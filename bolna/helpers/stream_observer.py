"""Optional sink for partial turns, so a host process can stream a live transcript.

Bolna commits a turn to ConversationHistory once: the user's when the transcript is
FINAL, the agent's when the response is complete. Anything watching that history —
our engine shim publishes it to ``transcript:{session_id}`` — therefore sees a whole
turn appear at once, several seconds after the speaker began. The partial text exists
in both cases (Deepgram interims, and the LLM's own token stream); it simply had
nowhere to go.

This module is that nowhere. Default is a no-op, so nothing changes for a caller that
does not opt in. ``set_observer`` is process-wide, matching how the shim already binds
per-call state through a ContextVar rather than through the engine's own plumbing.

The observer is called on the hot path of every turn, so it must not raise and must not
block: exceptions are swallowed here rather than at each call site, and an implementation
that needs to do I/O should hand it to the event loop rather than await it.
"""

from .logger_config import configure_logger

logger = configure_logger(__name__)

_observer = None


def set_observer(fn) -> None:
    """Install the sink. ``fn(role, content, key)`` — ``content`` is the text SO FAR,
    not a delta, so a consumer can render it without reassembling anything."""
    global _observer
    _observer = fn


def emit_partial(role: str, content: str, key=None) -> None:
    """Report the in-progress text of a turn. Safe to call when nobody is listening."""
    if _observer is None or not content:
        return
    try:
        _observer(role, content, key)
    except Exception as e:  # never let a watcher break the call it is watching
        logger.debug(f"stream observer failed: {e}")
