"""Synchronous, zero-dependency HTTP client for the KaptchaOCR captcha API server."""

from .client import Client
from .errors import KaptchaOcrError, NetworkError, ServerError
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

__version__ = "0.1.0"

__all__ = [
    "Client",
    "KaptchaOcrError",
    "NetworkError",
    "ServerError",
    "ClickBox",
    "ClickPoint",
    "ClickResult",
    "DetResult",
    "MathResult",
    "OcrResult",
    "SlideComparisonResult",
    "SlideMatchResult",
    "SlideResult",
    "SolveResult",
    "StatusResult",
    "TrajectoryPoint",
    "__version__",
]
