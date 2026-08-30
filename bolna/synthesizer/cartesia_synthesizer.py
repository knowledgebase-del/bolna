import asyncio
import json
import os
import time
import uuid

import aiohttp
import base64
from collections import deque
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context


logger = configure_logger(__name__)


class CartesiaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice_id,
        voice,
        language="en",
        model="sonic-english",
        audio_format="mp3",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        caching=True,
        speed=1.0,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # Cartesia always streams
            provider_name="cartesia",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["CARTESIA_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.language = language
        self.sampling_rate = sampling_rate
        self.speed = speed
        self.use_mulaw = kwargs.get("use_mulaw", True)  # web/freeswitch pass False → raw PCM @sampling_rate
        self.stream = True

        self.cartesia_host = os.getenv("CARTESIA_API_HOST", "api.cartesia.ai")
        self.ws_url = f"wss://{self.cartesia_host}/tts/websocket?api_key={self.api_key}&cartesia_version=2024-06-10"
        self.api_url = f"https://{self.cartesia_host}/tts/bytes"

        # Context tracking for interruption
        self.context_id = None
        self.turn_id = 0
        self.sequence_id = 0
        self.context_ids_to_ignore = set()
        # Word timings not yet attributed to an audio chunk, and how much audio has
        # been emitted for the current context. Cartesia sends timestamps as it
        # generates — generally ahead of the audio they describe — so they queue here
        # until the chunk that covers them goes out.
        self._pending_words = deque()
        self._audio_elapsed_s = 0.0
        self.ws_request_id = None
        # end_of_llm_stream sent on the current context → the next push must open a fresh one.
        self.context_finalized = False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        # WS always requests pcm_mulaw@8kHz (see form_payload)
        return "mulaw"

    def _on_push(self, meta_info, text):
        """Fresh context_id on turn/sequence change or after the previous finalized. The handoff
        doesn't finalize — the reply continues its context, avoiding parallel-context overlap."""
        if meta_info.get("message_category") == "handoff" and meta_info.get("end_of_llm_stream"):
            meta_info["end_of_llm_stream"] = False
        if not self.context_id:
            self._update_context(meta_info)
        elif self.context_finalized:
            self._update_context(meta_info)
        elif self.turn_id != meta_info.get("turn_id", 0) or self.sequence_id != meta_info.get("sequence_id", 0):
            self._update_context(meta_info)
        self.context_finalized = meta_info.get("end_of_llm_stream", False)

    def _update_context(self, meta_info):
        self.context_id = str(uuid.uuid4())
        # A new context is a new utterance: its audio clock restarts at zero, and any
        # words left over from the last one (a barge-in cancels mid-sentence) would
        # otherwise be attributed to this one's first chunk.
        self._pending_words.clear()
        self._audio_elapsed_s = 0.0
        self.turn_id = meta_info.get("turn_id", 0)
        self.sequence_id = meta_info.get("sequence_id", 0)
        logger.info(
            f"Cartesia new context_id={self.context_id} turn_id={self.turn_id} sequence_id={self.sequence_id} request_id={self.ws_request_id}"
        )

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {"context_id": self.context_id, "cancel": True}
                logger.info(f"handle_interruption: {interrupt_message}")
                await self.websocket.send(json.dumps(interrupt_message))
                self.context_id = None
                # The interrupted context's end-of-stream is now dropped, so the
                # next turn must be re-detected as new to clear stale queue entries.
                self.current_turn_start_time = None
        except Exception as e:
            logger.error(f"Error in handle_interruption: {e}")

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def form_payload(self, text):
        payload = {
            "context_id": self.context_id,
            "model_id": self.model,
            "transcript": text,
            "language": self.language,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": (
                {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000}
                if self.use_mulaw
                else {"container": "raw", "encoding": "pcm_s16le", "sample_rate": int(self.sampling_rate)}
            ),
            "generation_config": {"speed": self.speed},
            # Word-level timings, so each audio chunk can be labelled with the words it
            # actually speaks (meta_info["text_synthesized"]). Without this the stream
            # yields audio and nothing else, every mark carries text_len=0, and a client
            # can only show the sentence at LLM pace — seconds before the voice reaches
            # it. If a model ignores or rejects the flag no timestamps arrive, the text
            # stays empty, and everything behaves exactly as it did before.
            "add_timestamps": True,
        }
        if text:
            payload["continue"] = True
        return payload

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            await self._wait_for_ws()

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    payload = self.form_payload(text)
                    logger.info(
                        f"Cartesia sender context_id={self.context_id} text_len={len(text)} request_id={self.ws_request_id}"
                    )
                    await self._send_json(payload)
                except Exception as e:
                    logger.error(
                        f"Error sending chunk context_id={self.context_id} request_id={self.ws_request_id}: {e}"
                    )
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                logger.info(
                    f"Cartesia sender end_of_llm_stream context_id={self.context_id} request_id={self.ws_request_id}"
                )
                try:
                    await self._send_json(self.form_payload(""))
                except Exception as e:
                    logger.error(
                        f"Error sending end-of-stream signal context_id={self.context_id} request_id={self.ws_request_id}: {e}"
                    )
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    def _bytes_per_second(self) -> int:
        """Playback rate of the format we asked Cartesia for (see form_payload)."""
        return 8000 if self.use_mulaw else int(self.sampling_rate) * 2

    def _words_spoken_by(self, audio: bytes) -> str:
        """The words this chunk speaks, from the buffered timings.

        Each chunk covers the audio span [elapsed, elapsed + its own duration); every
        buffered word whose start falls inside that span is spoken by it. Words that
        have not arrived yet simply land on a later chunk — the reveal lags a beat and
        catches up, and the turn's last chunk settles the text either way.
        """
        bps = self._bytes_per_second()
        if not bps:
            return ""
        span_end = self._audio_elapsed_s + (len(audio) / bps)
        spoken = []
        while self._pending_words and self._pending_words[0][1] < span_end:
            spoken.append(self._pending_words.popleft()[0])
        self._audio_elapsed_s = span_end
        return (" ".join(spoken) + " ") if spoken else ""

    def _unpack_receiver_message(self, item):
        """receiver() yields (audio, {"text_synthesized": ...}) for audio chunks and
        bare bytes for the end-of-stream sentinel."""
        if isinstance(item, tuple):
            return item
        return item, {}

    async def receiver(self):
        not_connected_since = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("Cartesia receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if data.get("context_id") in self.context_ids_to_ignore:
                    continue

                # Word timings for the audio still to come. Buffered rather than used
                # immediately: they arrive ahead of the chunks that speak them.
                wt = data.get("word_timestamps")
                if wt and wt.get("words"):
                    words = wt.get("words") or []
                    starts = wt.get("start") or []
                    for i, w in enumerate(words):
                        self._pending_words.append((w, starts[i] if i < len(starts) else 0.0))
                    continue

                if "data" in data and data["data"]:
                    audio = base64.b64decode(data["data"])
                    yield (audio, {"text_synthesized": self._words_spoken_by(audio)})
                    continue
                elif "done" in data and data["done"]:
                    logger.info(
                        f"Cartesia recv done context_id={data.get('context_id')} request_id={self.ws_request_id}"
                    )
                    yield b"\x00"
                else:
                    logger.info(
                        f"No audio data in the response context_id={data.get('context_id')} request_id={self.ws_request_id}"
                    )

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url)), timeout=10.0
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            if hasattr(websocket, "response") and hasattr(websocket.response, "headers"):
                self.ws_request_id = websocket.response.headers.get("x-request-id")
                logger.info(
                    f"Cartesia WebSocket connected request_id={self.ws_request_id} connection_time={self.connection_time}ms"
                )
            else:
                logger.info(f"Cartesia WebSocket connected connection_time={self.connection_time}ms")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Cartesia websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Cartesia authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Cartesia endpoint not found: {e}")
            else:
                logger.error(f"Cartesia handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Cartesia: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP fallback (used by synthesize())
    # ------------------------------------------------------------------

    async def synthesize(self, text):
        return await self._generate_http(text)

    async def _generate_http(self, text):
        payload = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
            "language": self.language,
            "generation_config": {"speed": self.speed},
        }
        headers = {"X-API-Key": self.api_key, "Cartesia-Version": "2024-06-10"}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None
