import asyncio
import ipaddress
import json
import os
import re
import socket
from urllib.parse import quote, urlsplit

import aiohttp
from yarl import URL
from bolna.helpers.logger_config import configure_logger
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log, format_error_message

logger = configure_logger(__name__)

ALLOWED_URL_SCHEMES = ("http", "https")

# Hosts in BOLNA_TOOL_URL_HOST_ALLOWLIST (comma-separated) bypass the SSRF block,
# for deployments that legitimately call an internal endpoint from a tool.
_ALLOWLISTED_HOSTS = frozenset(
    h.strip().lower() for h in os.getenv("BOLNA_TOOL_URL_HOST_ALLOWLIST", "").split(",") if h.strip()
)


class SSRFError(ValueError):
    """Raised when an outbound request targets a non-public address."""


def _is_disallowed_ip(ip):
    """True if ``ip`` (an ``ipaddress`` object) is not safe to connect to."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # ``is_global`` is the authoritative "publicly routable" check; the explicit
    # flags are belt-and-suspenders across Python/ipaddress versions.
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local  # 169.254.0.0/16 (AWS/GCP metadata) and fe80::/10
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def validate_outbound_url(url):
    """SSRF guard for user-supplied URLs.

    Rejects non-http(s) schemes and any host that resolves to a non-public
    address (loopback, RFC-1918, link-local incl. the 169.254.169.254 cloud
    metadata endpoint, reserved, etc.). Raises ``SSRFError`` on a blocked URL.

    Resolution happens here rather than only checking the literal string so that
    a hostname pointing at an internal address is also rejected. (A short
    rebind-after-check window remains; closing it fully needs connect-time
    pinning, which we can layer on later.)
    """
    if not isinstance(url, str):
        raise SSRFError("Missing or invalid request URL")
    url = url.strip()
    if not url:
        raise SSRFError("Missing or invalid request URL")

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise SSRFError(f"Blocked URL scheme {scheme!r}; only http/https are allowed")

    host = parsed.hostname
    if not host:
        raise SSRFError("Request URL has no host")

    if host.lower() in _ALLOWLISTED_HOSTS:
        return

    try:
        port = parsed.port
    except ValueError:
        raise SSRFError("Request URL has an invalid port")

    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(loop.getaddrinfo(host, port, type=socket.SOCK_STREAM), timeout=5)
    except asyncio.TimeoutError:
        raise SSRFError(f"DNS resolution for host {host!r} timed out")
    except socket.gaierror as exc:
        raise SSRFError(f"Could not resolve host {host!r}: {exc}")

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise SSRFError(f"Could not validate resolved address {addr!r} for host {host!r}")
        if _is_disallowed_ip(ip):
            raise SSRFError(f"Blocked request to non-public address {addr} (resolved from {host})")


def _contains_var_markers(obj):
    """
    Check if object contains any {"$var": ...} markers.

    Args:
        obj: JSON object (dict, list, or primitive)

    Returns:
        True if $var markers are found, False otherwise
    """
    if isinstance(obj, dict):
        if "$var" in obj:
            return True
        return any(_contains_var_markers(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(_contains_var_markers(item) for item in obj)
    return False


def substitute_var_markers(obj, values):
    """
    Recursively substitute {"$var": "name"} markers with actual values.

    This provides type-safe JSON substitution where arrays remain arrays
    and objects remain objects, without string manipulation.

    Args:
        obj: JSON object (dict, list, or primitive)
        values: dict of variable names to values

    Returns:
        Object with markers replaced by actual values

    Example:
        obj = {"products": {"$var": "products"}, "static": "value"}
        values = {"products": [{"code": "123"}]}
        result = {"products": [{"code": "123"}], "static": "value"}
    """
    if isinstance(obj, dict):
        # Check if this is a $var marker
        if len(obj) == 1 and "$var" in obj:
            var_name = obj["$var"]
            if var_name in values:
                return values[var_name]
            else:
                # Keep marker if no value provided (for debugging)
                logger.warning(f"No value provided for $var marker: {var_name}")
                return obj
        # Recursively process dict values
        return {k: substitute_var_markers(v, values) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_var_markers(item, values) for item in obj]
    else:
        return obj  # Primitives returned as-is


# A {{name}} template in a URL or a header value.
#
# Two substitution syntaxes exist here, for two different jobs, and neither can do
# the other's. A request BODY uses {"$var": "name"} markers, because a body is typed
# JSON and a marker can carry a list or an object through intact. A URL or a header
# is a string that usually needs the value embedded in other text —
# "Bearer {{access_token}}", "/v1/patients/{{patient_id}}/appointments" — which a
# marker cannot express. Double braces because that is what the portal's fields
# already contain, so this is authorable without any UI change.
_STRING_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")

# A tool may override the default request timeout, but only within reason: longer
# than this and the caller is listening to silence while the call clock runs.
_DEFAULT_TIMEOUT_SECONDS = 10
_MAX_TIMEOUT_SECONDS = 120


def substitute_templates(text: str, values: dict, quote_value: bool = False):
    """``(substituted text, names that had no value)``.

    ``quote_value`` percent-encodes what it inserts, for URLs — a value carrying a
    ``/`` or a ``?`` would otherwise change the shape of the request rather than
    fill a slot in it.
    """
    missing = []

    def replace(match):
        name = match.group(1)
        if name not in values:
            missing.append(name)
            return match.group(0)
        value = str(values[name])
        return quote(value, safe="") if quote_value else value

    return _STRING_TEMPLATE.sub(replace, text), missing


def resolve_timeout(timeout_ms) -> int:
    """A tool's own request timeout in seconds, clamped, defaulting to 10."""
    try:
        seconds = int(timeout_ms) / 1000.0
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    if seconds <= 0:
        return _DEFAULT_TIMEOUT_SECONDS
    return int(min(seconds, _MAX_TIMEOUT_SECONDS))


def unresolved_var_markers(obj) -> list:
    """Names of ``$var`` markers still present after substitution."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "$var" in obj:
            return [obj["$var"]]
        return [name for v in obj.values() for name in unresolved_var_markers(v)]
    if isinstance(obj, list):
        return [name for item in obj for name in unresolved_var_markers(item)]
    return []


def prepare_api_request(param, api_token, headers_data, context_values=None, **kwargs):
    """Build a tool's outbound request.

    ``kwargs`` are the model's own function-call arguments. ``context_values`` are
    the call's variables — what the caller passed in when placing the call, plus
    everything captured since (tool responses, extracted variables). A ``$var``
    marker resolves against both, with the model's argument winning a name clash
    since it is the more specific of the two.

    Passed as one named dict rather than merged into ``kwargs`` on purpose: the
    caller splats the model's response into ``trigger_api``, so a captured
    variable named ``url`` or ``method`` would otherwise collide with that
    function's own parameters and take the tool call down.
    """
    request_body, api_params = None, None
    if param:
        # NEW FORMAT: Check for $var markers (type-safe JSON substitution)
        if isinstance(param, dict) and _contains_var_markers(param):
            api_params = substitute_var_markers(param, {**(context_values or {}), **kwargs})
            missing = unresolved_var_markers(api_params)
            if missing:
                # Sending the marker object verbatim would POST {"$var": "dob"} where
                # the API expects a string — silently malformed data reaching a real
                # system. Raising instead surfaces as a normal tool error the model
                # can route on (trigger_api catches it and returns an error body).
                raise ValueError(
                    f"tool call is missing {len(missing)} value(s) with no variable to "
                    f"resolve from: {', '.join(sorted(set(missing)))}"
                )
            request_body = json.dumps(api_params)
            logger.info("Using $var marker substitution for param")
        else:
            # LEGACY FORMAT: String template with %(field)s placeholders
            if isinstance(param, dict):
                param = json.dumps(param)

            # JSON-serialize complex values (lists, dicts) for proper string substitution
            # Python's % formatting uses repr() which produces single quotes,
            # but JSON requires double quotes for valid JSON output
            json_kwargs = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in kwargs.items()}

            request_body = param % json_kwargs
            api_params = json.loads(request_body)

    headers = {"Content-Type": "application/json"}
    content_type = "json"
    if api_token:
        headers["Authorization"] = api_token

    if headers_data and isinstance(headers_data, dict):
        for k, v in headers_data.items():
            headers[k] = v

    if headers.get("Content-Type").lower().startswith("application/x-www-form-urlencoded"):
        content_type = "form"

    return {
        "request_body": request_body,
        "api_params": api_params,
        "headers": headers,
        "content_type": content_type,
    }


def build_get_url(url, api_params):
    """Assemble the final GET URL, treating the base url as already in wire form.

    The base url is sent verbatim (encoded=True) so a pre-encoded value such as a nested
    callback URL (e.g. an Ozonetel appURL) keeps its %3F/%3A/%2F instead of being decoded
    to a literal ?/:/ that receivers truncate at. Param values are percent-encoded once and
    appended, so plain values are unchanged and any param that is itself a URL is encoded
    correctly rather than left with a query-splitting literal '?'.
    """
    final_url = url
    if api_params:
        sep = "&" if "?" in url else "?"
        final_url = (
            url + sep + "&".join(f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in api_params.items())
        )
    return URL(final_url, encoded=True)


async def trigger_api(
    url, method, param, api_token, headers_data, meta_info, run_id, return_response_metadata=False,
    context_values=None, timeout_ms=None, **kwargs
):
    timeout_seconds = resolve_timeout(timeout_ms)
    try:
        # {{name}} in the URL and in header values, from the same pool the body
        # resolves against: the call's variables, with the model's own arguments
        # winning a clash.
        #
        # SUBSTITUTE FIRST, VALIDATE SECOND. validate_outbound_url is the SSRF guard,
        # and a template can carry a host — a tool response feeding {{host}} into
        # the URL would sail straight past a check run on the un-substituted string.
        values = {**(context_values or {}), **kwargs}
        url, missing = substitute_templates(url, values, quote_value=True)
        if isinstance(headers_data, dict):
            substituted = {}
            for key, value in headers_data.items():
                if isinstance(value, str):
                    value, header_missing = substitute_templates(value, values)
                    missing += header_missing
                substituted[key] = value
            headers_data = substituted
        if missing:
            # Same rule as an unresolved $var in the body: sending "Bearer {{token}}"
            # verbatim is a request that fails somewhere less obvious.
            raise ValueError(
                f"tool call has {len(missing)} unresolved value(s) in its URL or headers: "
                f"{', '.join(sorted(set(missing)))}"
            )

        await validate_outbound_url(url)
        prepared_request = prepare_api_request(
            param, api_token, headers_data, context_values=context_values, **kwargs
        )
        request_body = prepared_request["request_body"]
        api_params = prepared_request["api_params"]
        headers = prepared_request["headers"]
        content_type = prepared_request["content_type"]
        convert_to_request_log(
            request_body,
            meta_info,
            None,
            LogComponent.FUNCTION_CALL,
            direction=LogDirection.REQUEST,
            is_cached=False,
            run_id=run_id,
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
            response = None
            response_text = None
            if method.lower() == "get":
                get_url = build_get_url(url, api_params)
                logger.info(f"Sending request {request_body}, {get_url}, {headers}")
                # allow_redirects=False: the URL is validated pre-flight, but a redirect
                # hop is not re-validated and would reopen the SSRF path (e.g. 302 -> IMDS).
                async with session.get(get_url, headers=headers, allow_redirects=False) as response:
                    response_text = await response.text()
            elif method.lower() == "post":
                logger.info(f"Sending request {api_params}, {url}, {headers}")
                if content_type == "json":
                    async with session.post(url, json=api_params, headers=headers, allow_redirects=False) as response:
                        response_text = await response.text()
                elif content_type == "form":
                    normalized_api_params = normalize_for_form(api_params)
                    async with session.post(
                        url, data=normalized_api_params, headers=headers, allow_redirects=False
                    ) as response:
                        response_text = await response.text()
                else:
                    raise ValueError(
                        f"Unsupported Content-Type for POST: {headers.get('Content-Type')!r}. "
                        "Only 'application/json' and 'application/x-www-form-urlencoded' are supported."
                    )
            else:
                raise ValueError(f"Unsupported HTTP method: {method!r}. Only 'GET' and 'POST' are supported.")

            if response is not None:
                logger.info(f"Final URL: {response.url}")

            if return_response_metadata:
                return {
                    "status_code": response.status if response is not None else None,
                    "body": response_text if response_text is not None else "",
                    "content_type": response.headers.get("Content-Type") if response is not None else None,
                }

            return response_text if response_text is not None else ""
    except SSRFError as e:
        # Log the full reason server-side, but return a generic message: the resolved
        # IP/host in the SSRFError text would otherwise flow back to the LLM and hand a
        # non-blind SSRF probe the exact internal address it was trying to discover.
        logger.warning(f"Blocked outbound request to {url}: {e}")
        message = "ERROR CALLING API: request blocked by outbound URL policy"
        if run_id:
            convert_to_request_log(
                format_error_message("function_call", url, "blocked by outbound URL policy"),
                meta_info,
                model=None,
                component=LogComponent.WARNING,
                direction=LogDirection.WARNING,
                is_cached=False,
                run_id=run_id,
            )
        if return_response_metadata:
            return {
                "status_code": None,
                "body": message,
                "content_type": None,
                "error": "blocked by outbound URL policy",
            }
        return message
    except asyncio.TimeoutError:
        message = f"ERROR CALLING API: Request to {url} timed out after {timeout_seconds} seconds"
        logger.debug(message)
        if run_id:
            convert_to_request_log(
                format_error_message("function_call", url, f"Timed out after {timeout_seconds} seconds"),
                meta_info,
                model=None,
                component=LogComponent.WARNING,
                direction=LogDirection.WARNING,
                is_cached=False,
                run_id=run_id,
            )
        if return_response_metadata:
            return {
                "status_code": None,
                "body": message,
                "content_type": None,
                "error": f"Timed out after {timeout_seconds} seconds",
            }
        return message
    except Exception as e:
        message = f"ERROR CALLING API: Please check your API: {e}"
        logger.debug(message)
        if run_id:
            convert_to_request_log(
                format_error_message("function_call", url, str(e)),
                meta_info,
                model=None,
                component=LogComponent.WARNING,
                direction=LogDirection.WARNING,
                is_cached=False,
                run_id=run_id,
            )
        if return_response_metadata:
            return {
                "status_code": None,
                "body": message,
                "content_type": None,
                "error": str(e),
            }
        return message


async def computed_api_response(response):
    get_res_keys, get_res_values = None, None
    try:
        get_res_keys = list(json.loads(response).keys())
        get_res_values = list(json.loads(response).values())
    except Exception as e:
        pass

    return get_res_keys, get_res_values


def normalize_for_form(data: dict) -> dict:
    return {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
