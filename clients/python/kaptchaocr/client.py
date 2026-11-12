"""Synchronous, zero-dependency HTTP client for the KaptchaOCR captcha API server.

Only the Python standard library is used, so the client can be vendored into
scraper projects without additional dependencies.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from .errors import NetworkError, ServerError
from .models import (
    ClickBox,
    ClickPoint,
    ClickResult,
    DetResult,
    MathResult,
    OcrResult,
    SlideComparisonResult,
    SlideMatchResult,
    SlideResult,
    SolveResult,
    StatusResult,
    TrajectoryPoint,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

ImageLike = Union[bytes, bytearray, memoryview, str, os.PathLike]


class TransportResponse(NamedTuple):
    """Raw response returned by a transport."""

    status: int
    headers: Dict[str, str]
    body: bytes


#: `(method, url, headers, body, timeout) -> TransportResponse`
Transport = Callable[..., TransportResponse]


def _urlopen_transport(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[bytes],
    timeout: float,
) -> TransportResponse:
    """Default transport built on `urllib.request`."""

    request = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return TransportResponse(response.status, response_headers, response.read())
    except urllib.error.HTTPError as exc:
        response_headers = {key.lower(): value for key, value in (exc.headers or {}).items()}
        return TransportResponse(exc.code, response_headers, exc.read())


def _as_bytes(image: ImageLike) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)

    if isinstance(image, (str, os.PathLike)):
        with open(os.fspath(image), "rb") as handle:
            return handle.read()

    raise TypeError(f"image must be bytes or a path, got {type(image).__name__}")


def _retry_after_seconds(headers: Optional[Dict[str, str]]) -> Optional[float]:
    if not headers:
        return None

    value = headers.get("retry-after")

    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


class Client:
    """Synchronous client for one KaptchaOCR server.

    :param base_url: server address, e.g. ``http://127.0.0.1:8000``.
    :param api_key: optional key sent as ``x-api-key``.
    :param timeout: per-request timeout in seconds.
    :param retries: extra attempts after the first one for 5xx / 429 /
        connection errors (exponential backoff, ``Retry-After`` respected).
    :param transport: override the HTTP transport (mainly for tests).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        retries: int = 2,
        transport: Optional[Transport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self._transport: Transport = transport or _urlopen_transport

    # -- high-level API ---------------------------------------------------

    def solve(
        self,
        image: ImageLike,
        *,
        math: str = "auto",
        png_fix: Optional[bool] = None,
        probability: Optional[bool] = None,
        charset_range: Optional[str] = None,
        color_filter: Any = None,
        case: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> SolveResult:
        """Recognize a captcha and (with ``math="auto"``) solve it as an
        arithmetic expression."""

        payload = self._image_payload(
            image,
            math=math,
            png_fix=png_fix,
            probability=probability,
            charset_range=charset_range,
            color_filter=color_filter,
            case=case,
            min_confidence=min_confidence,
        )
        data = self._request("POST", "/solve", payload)
        math_data = data.get("math")

        return SolveResult(
            text=data.get("text", ""),
            math=MathResult(
                expression=math_data["expression"],
                result=math_data["result"],
                answer=math_data["answer"],
            )
            if math_data
            else None,
            probability=data.get("probability"),
            charset=data.get("charset"),
            confidence=data.get("confidence"),
            model=data.get("model"),
            raw=data,
        )

    def solve_file(self, path: ImageLike, **kwargs: Any) -> SolveResult:
        """Like :meth:`solve`, but reads the image from a file path."""

        return self.solve(path, **kwargs)

    def ocr(
        self,
        image: ImageLike,
        *,
        png_fix: Optional[bool] = None,
        probability: Optional[bool] = None,
        charset_range: Optional[str] = None,
        color_filter: Any = None,
        case: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> OcrResult:
        """Recognize a captcha, returning only the text (and optional
        probability data)."""

        payload = self._image_payload(
            image,
            png_fix=png_fix,
            probability=probability,
            charset_range=charset_range,
            color_filter=color_filter,
            case=case,
            min_confidence=min_confidence,
        )
        data = self._request("POST", "/ocr", payload)

        return OcrResult(
            text=data.get("text", ""),
            probability=data.get("probability"),
            charset=data.get("charset"),
            confidence=data.get("confidence"),
            raw=data,
        )

    def det(self, image: ImageLike) -> DetResult:
        """Run object detection, returning boxes and scores."""

        data = self._request("POST", "/det", {"image": self._encode(image)})

        return DetResult(
            bboxes=data.get("bboxes", []),
            scores=data.get("scores", []),
            raw=data,
        )

    def solve_click(
        self,
        image: ImageLike,
        prompt: Optional[str] = None,
        *,
        score_threshold: Optional[float] = None,
        charset_range: Optional[str] = None,
    ) -> ClickResult:
        """Solve a click captcha: detect glyph boxes, recognize each box and
        order the click points for ``prompt`` (e.g. ``请依次点击【明,天,会】``
        or ``click in order: A B C``).

        Without a prompt only the recognized boxes are returned (``points`` is
        empty and ``prompt`` is ``None``)."""

        payload: Dict[str, Any] = {"image": self._encode(image)}

        if prompt is not None:
            payload["prompt"] = prompt

        if score_threshold is not None:
            payload["score_threshold"] = score_threshold

        if charset_range is not None:
            payload["charset_range"] = charset_range

        data = self._request("POST", "/solve/click", payload)

        points = [
            ClickPoint(x=float(point["x"]), y=float(point["y"]))
            for point in data.get("points") or []
        ]
        boxes = [
            ClickBox(
                x1=int(box["x1"]),
                y1=int(box["y1"]),
                x2=int(box["x2"]),
                y2=int(box["y2"]),
                ch=str(box.get("ch", "")),
                score=float(box.get("score", 0.0)),
            )
            for box in data.get("boxes") or []
        ]

        return ClickResult(
            prompt=data.get("prompt"),
            points=points,
            boxes=boxes,
            unmatched=list(data.get("unmatched") or []),
            raw=data,
        )

    def slide_match(
        self,
        target_image: ImageLike,
        background_image: ImageLike,
        *,
        simple_target: Optional[bool] = None,
    ) -> SlideMatchResult:
        """Find the slider gap position."""

        payload: Dict[str, Any] = {
            "target_image": self._encode(target_image),
            "background_image": self._encode(background_image),
        }

        if simple_target is not None:
            payload["simple_target"] = simple_target

        data = self._request("POST", "/slide_match", payload)

        return SlideMatchResult(
            target=data.get("target", []),
            target_x=data.get("target_x", 0),
            target_y=data.get("target_y", 0),
            raw=data,
        )

    def slide_comparison(
        self,
        target_image: ImageLike,
        background_image: ImageLike,
    ) -> SlideComparisonResult:
        """Find the gap position by comparing the two images."""

        payload = {
            "target_image": self._encode(target_image),
            "background_image": self._encode(background_image),
        }
        data = self._request("POST", "/slide_comparison", payload)

        return SlideComparisonResult(x=data.get("x", 0), y=data.get("y", 0), raw=data)

    def slide_solve(
        self,
        target_image: ImageLike,
        background_image: ImageLike,
        *,
        simple_target: Optional[bool] = None,
        handle_x: Optional[float] = None,
        handle_y: Optional[float] = None,
        offset: Optional[float] = None,
        duration_ms: Optional[float] = None,
        seed: Optional[int] = None,
        trajectory: Optional[bool] = None,
    ) -> SlideResult:
        """Find the gap and return a drag distance plus a human-like
        trajectory (deltas from the handle start)."""

        payload: Dict[str, Any] = {
            "target_image": self._encode(target_image),
            "background_image": self._encode(background_image),
        }

        for key, value in (
            ("simple_target", simple_target),
            ("handle_x", handle_x),
            ("handle_y", handle_y),
            ("offset", offset),
            ("duration_ms", duration_ms),
            ("seed", seed),
            ("trajectory", trajectory),
        ):
            if value is not None:
                payload[key] = value

        data = self._request("POST", "/slide/solve", payload)
        points = [
            TrajectoryPoint(x=float(p["x"]), y=float(p["y"]), t=float(p["t"]))
            for p in data.get("trajectory") or []
        ]

        return SlideResult(
            target=data.get("target", []),
            target_x=data.get("target_x", 0),
            target_y=data.get("target_y", 0),
            gap_x=data.get("gap_x", 0),
            gap_y=data.get("gap_y", 0),
            distance=float(data.get("distance", 0.0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            trajectory=points,
            raw=data,
        )

    def status(self) -> StatusResult:
        """Return the server status and enabled features."""

        data = self._request("GET", "/status")

        return StatusResult(
            service_status=data.get("service_status", ""),
            enabled_features=data.get("enabled_features", []),
            raw=data,
        )

    def healthz(self) -> bool:
        """Liveness probe: ``True`` when the process answers ``/healthz``."""

        try:
            response = self._transport("GET", f"{self.base_url}/healthz", {}, None, self.timeout)
        except Exception:
            return False

        return response.status == 200

    # -- internals --------------------------------------------------------

    def _image_payload(self, image: ImageLike, **fields: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"image": self._encode(image)}

        for key, value in fields.items():
            if value is not None:
                payload[key] = value

        return payload

    @staticmethod
    def _encode(image: ImageLike) -> str:
        return base64.b64encode(_as_bytes(image)).decode("ascii")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["x-api-key"] = self.api_key

        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        url = f"{self.base_url}{path}"
        attempt = 0

        while True:
            try:
                response = self._transport(method, url, self._headers(), body, self.timeout)
            except Exception as exc:  # urllib.error.URLError, socket.timeout, OSError ...
                if attempt < self.retries:
                    self._sleep(attempt, None)
                    attempt += 1
                    continue

                raise NetworkError(f"{method} {path} failed: {exc}") from exc

            try:
                envelope = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ServerError(response.status, response.status, "invalid JSON response") from exc

            code = int(envelope.get("code", response.status))

            if 200 <= response.status < 300 and code == 200:
                data = envelope.get("data")

                return data if isinstance(data, dict) else {}

            # Some older servers report 429 in the envelope while returning 200.
            # Retry that, but never retry a deterministic 4xx (bad input/auth).
            retryable = (
                response.status == 429
                or response.status >= 500
                or code == 429
            )

            if retryable and attempt < self.retries:
                self._sleep(attempt, response.headers)
                attempt += 1
                continue

            raise ServerError(response.status, code, str(envelope.get("msg", "request failed")))

    def _sleep(self, attempt: int, headers: Optional[Dict[str, str]]) -> None:
        delay = min(0.2 * (2**attempt), 5.0)
        retry_after = _retry_after_seconds(headers)

        if retry_after is not None:
            delay = min(retry_after, 5.0)

        time.sleep(delay)
