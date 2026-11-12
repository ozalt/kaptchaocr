"""Typed results returned by the client."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MathResult:
    """Arithmetic answer parsed from a math captcha."""

    expression: str
    result: float
    answer: str


@dataclass(frozen=True)
class SolveResult:
    """Result of `POST /solve`."""

    text: str
    math: Optional[MathResult]
    probability: Optional[List[List[float]]]
    charset: Optional[List[str]]
    confidence: Optional[float]
    model: Optional[str]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class OcrResult:
    """Result of `POST /ocr`."""

    text: str
    probability: Optional[List[List[float]]]
    charset: Optional[List[str]]
    confidence: Optional[float]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class DetResult:
    """Result of `POST /det`."""

    bboxes: List[List[int]]
    scores: List[float]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class TrajectoryPoint:
    """One sampled slider-drag point (deltas from the handle start)."""

    x: float
    y: float
    t: float


@dataclass(frozen=True)
class ClickBox:
    """One detected glyph box with its recognized character."""

    x1: int
    y1: int
    x2: int
    y2: int
    ch: str
    score: float

    @property
    def center(self) -> Tuple[float, float]:
        """Box centre in image coordinates."""

        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class ClickPoint:
    """A click coordinate relative to the captcha image."""

    x: float
    y: float


@dataclass(frozen=True)
class ClickResult:
    """Result of `POST /solve/click`."""

    prompt: Optional[str]
    points: List[ClickPoint]
    boxes: List[ClickBox]
    unmatched: List[str]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class SlideResult:
    """Result of `POST /slide/solve`."""

    target: List[int]
    target_x: int
    target_y: int
    gap_x: int
    gap_y: int
    distance: float
    duration_ms: float
    trajectory: List[TrajectoryPoint]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class SlideMatchResult:
    """Result of `POST /slide_match`."""

    target: List[int]
    target_x: int
    target_y: int
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class SlideComparisonResult:
    """Result of `POST /slide_comparison`."""

    x: int
    y: int
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class StatusResult:
    """Result of `GET /status`."""

    service_status: str
    enabled_features: List[str]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)
