# Python client

`clients/python/` contains a zero-dependency (standard library only) synchronous
client for a running KaptchaOCR server. The package is `kaptchaocr`.

## Install

Until the package is published to PyPI, install from the repository:

```sh
pip install clients/python

# editable, for development
pip install -e clients/python
```

## Quick start

```python
from kaptchaocr import Client

client = Client("http://127.0.0.1:8000", timeout=10, retries=2)

# server started with --ocr
result = client.solve(open("captcha.png", "rb").read())
print(result.text)              # e.g. "5+?=8"
if result.math:
    print(result.math.answer)   # "3"
```

## Methods

| Method | Server flags | Returns |
| --- | --- | --- |
| `ocr(image)` | `--ocr` | `OcrResult(text, probability, charset, confidence)` |
| `solve(image)` / `solve_file(path)` | `--ocr` | `SolveResult(text, math, ...)` |
| `solve_click(image, prompt)` | `--ocr --det` | `ClickResult(prompt, points, boxes, unmatched)` |
| `det(image)` | `--det` | `DetResult(bboxes, scores)` |
| `slide_match(target, background, simple_target=False)` | `--slide` | `SlideMatchResult` |
| `slide_comparison(target, background)` | `--slide` | `SlideComparisonResult` |
| `slide_solve(target, background, handle_x=..., handle_y=...)` | `--slide` | `SlideSolveResult(distance, trajectory, ...)` |
| `status()` | always | `StatusResult(service_status, enabled_features)` |
| `healthz()` | always | `bool` |

Images may be `bytes`, `bytearray`, `memoryview`, or a file path in the
`*_file` helpers.

```python
slide = client.slide_solve(target_bytes, background_bytes, handle_x=12, handle_y=40)
print(slide.distance, slide.duration_ms)
for point in slide.trajectory:
    ...  # mouse.move(handle_x + point.x, handle_y + point.y)

click = client.solve_click(image_bytes, "请依次点击【明,天】")
for point in click.points:
    ...  # relative to the captcha image
```

## Errors and retries

- `ServerError(status, code, msg)` — HTTP or business error; 4xx is not retried.
- `NetworkError` — the request never reached the server after retries.
- Retried automatically: connection errors, HTTP 5xx and 429, with exponential
  backoff (`0.2 * 2^n`, capped at 5 s); `Retry-After` is honored.

Pass `api_key="..."` when the server runs with `--api-key`.

## Tests and examples

```sh
cd clients/python
python3 -m unittest discover -v
```

Playwright examples: `clients/python/examples/playwright_text.py`,
`playwright_slider.py`, `playwright_click.py`. Full client notes:
[`clients/python/README.md`](../clients/python/README.md).
