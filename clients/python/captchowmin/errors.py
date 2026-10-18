"""Exceptions raised by the captchowmin client."""


class CaptchowminError(Exception):
    """Base class for all client errors."""


class ServerError(CaptchowminError):
    """The server answered with an HTTP or business error code."""

    def __init__(self, status: int, code: int, msg: str):
        super().__init__(f"server returned HTTP {status} (code {code}): {msg}")
        self.status = status
        self.code = code
        self.msg = msg


class NetworkError(CaptchowminError):
    """The request never reached the server (connection error or timeout)."""
