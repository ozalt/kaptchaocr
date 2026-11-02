# KaptchaOCR Python client

Zero-dependency (standard library only) synchronous client for a running
[KaptchaOCR](../../README.md) API server.

## Install

```sh
# from the repository root
pip install clients/python

# or editable, for development
pip install -e clients/python
```

## Quickstart

```python
from kaptchaocr import Client

client = Client("http://127.0.0.1:8000", timeout=10, retries=2)

# Text / math captcha (server started with --ocr)
result = client.solve(open("captcha.png", "rb").read())
print(result.text)                        # raw OCR text, e.g. "5+?=8"
if result.math:
    print(result.math.answer)             # "3"

# Slider captcha (server started with --slide)
slide = client.slide_solve(target_bytes, background_bytes, handle_x=12, handle_y=40)
print(slide.distance, slide.duration_ms)
for point in slide.trajectory:
    ...                                   # mouse.move(handle_x + point.x, handle_y + point.y)

# Click captcha (server started with --ocr --det)
click = client.solve_click(image_bytes, "请依次点击【明,天】")
print(click.prompt, click.unmatched)
for point in click.points:
    ...                                   # relative to the captcha image
for box in click.boxes:
    ...                                   # box.ch, box.score, box.center

# Primitives
client.ocr(image_bytes)                   # OcrResult
client.det(image_bytes)                   # DetResult(bboxes, scores)
client.slide_match(target, background)    # SlideMatchResult
client.slide_comparison(target, background)
client.status()                           # StatusResult(service_status, enabled_features)
client.healthz()                          # bool
```

Images may be `bytes`, `bytearray`, `memoryview` or a file path; `solve_file`
exists for symmetry with `solve`.

## Errors and retries

- `ServerError(status, code, msg)` — HTTP or business error (4xx is never
  retried).
- `NetworkError` — the request never reached the server (connection error or
  timeout) after the configured retries.
- Retried automatically: connection errors, HTTP 5xx and 429, with
  exponential backoff (`0.2 * 2^n`, capped at 5 s); `Retry-After` is honored.

## Tests

No server is needed (the transport is faked):

```sh
cd clients/python
python3 -m unittest discover -v
```

## Examples

- `examples/playwright_text.py` — fetch the captcha screenshot, solve, fill,
  submit, retry with a fresh image.
- `examples/playwright_slider.py` — drag the handle along the returned
  trajectory.
- `examples/playwright_click.py` — read the prompt from the page, call
  `solve_click`, map the image coordinates onto the element and click.
