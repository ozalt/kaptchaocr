# Playwright Cookbook — captcha solving with kaptchowmin

Practical recipes for driving login / registration forms with
[Playwright](https://playwright.dev/python/) while the captcha work is
delegated to a local kaptchowmin server.

All snippets use the zero-dependency Python client from `clients/python/`
(`pip install clients/python`) and Playwright's sync API
(`pip install playwright && playwright install chromium`).

## 0. Setup

Terminal 1 — the server:

```sh
# all features, balanced fine-tuned recognition model (ft-small, ~80% exact)
./kaptchowmin --address 127.0.0.1:8000 --ocr --det --slide \
    --manifest model/models.json --profile balanced
```

Which recognition model should I use?

| captcha kind | model | why |
| --- | --- | --- |
| alphanumeric / case-sensitive | `--profile balanced` (ft-small) | fine-tuned on the 73-class captcha alphabet, fastest accurate option |
| need maximum accuracy | `--profile accurate` (ft-medium) | ~91% exact, ~2x slower |
| Chinese text / Chinese math (`九乘六等于？`) | `--rec-model stock-common` | the fine-tuned tiers only know the alphanumeric alphabet |

Terminal 2 — the browser part. Create the client once and reuse it:

```python
from captchowmin import Client

client = Client("http://127.0.0.1:8000", timeout=10, retries=2)
```

## 1. Text captchas

Run the full example in
[`clients/python/examples/playwright_text.py`](clients/python/examples/playwright_text.py).
The important parts:

- **Screenshot the captcha element, not the page**:
  `page.locator("#captcha-img").screenshot()`.
- **Wait until the image is really loaded** before screenshotting, otherwise
  you send a half-rendered GIF/spinner:
  ```python
  page.locator("#captcha-img").wait_for(state="visible")
  page.wait_for_load_state("networkidle")
  ```
- **Restrict the character set when you know it** — fewer lookalikes and
  better accuracy:
  ```python
  result = client.solve(image, charset_range="0123456789")
  ```
- **Gate on confidence instead of submitting garbage** (only when you ask for
  probabilities):
  ```python
  result = client.solve(image, probability=True)
  if result.confidence is not None and result.confidence < 0.80:
      ...  # reload for a fresh captcha instead of submitting
  ```
- **One attempt = one fresh image.** Captchas are usually invalidated after a
  failed submit, so reload the image (not just retry the POST). Cap attempts
  (3–5) and then fail the task.

## 2. Math captchas

The same `/solve` call answers arithmetic captchas; type `math.answer`:

```python
result = client.solve(image)                 # e.g. text="5+?=8"
value = result.math.answer if result.math else result.text
page.fill("#captcha", value)                 # -> "3"
```

- `3+5=`, `9÷3`, `4x6`, full-width `４×６＝`, and unknown forms
  (`5+?=8`, `?×6=42`, `12/?=4`) are all supported.
- Chinese math works with the stock model:
  `九乘六等于？` → `"54"`, `十二减四等于几` → `"8"`, `五加几等于八` → `"3"`.
  The answer is always returned in digits.
- If a site wants the whole expression plus the result (rare), compose it
  yourself from `text` and `answer`; check the input placeholder first.
- `math="off"` disables the solver and behaves like `/ocr`.

## 3. Slider captchas

Run the full example in
[`clients/python/examples/playwright_slider.py`](clients/python/examples/playwright_slider.py).
Key points:

- Pass the handle centre from the DOM as `handle_x` / `handle_y`; the
  trajectory is returned as **deltas**, so replay it as
  `mouse.move(handle_x + p.x, handle_y + p.y)` with ~16 ms between points:

  ```python
  page.mouse.move(handle_x, handle_y)
  page.mouse.down()
  for point in result.trajectory:
      page.mouse.move(handle_x + point.x, handle_y + point.y)
      page.wait_for_timeout(16)
  page.mouse.up()
  ```

- **Calibrate `offset`** while tuning: many sites render the piece a few pixels
  inland of the gap. Fix `seed=7` during calibration so the trajectory is
  reproducible, then drop it.
- Do not jump the mouse straight to the target — the eased path + tremor exists
  precisely to look human. Keep the per-point waits.
- Some sliders validate a moment after release; wait for the success indicator
  before submitting the form.

## 3b. Click-text captchas

Server flags: `--ocr --det`. `POST /solve/click` detects glyph boxes,
recognizes each character and returns the click order for the prompt text
(`请依次点击【明,天,会】`, `click in order: A B C`, ...).

```python
prompt = page.locator("#captcha-tip").inner_text()
captcha = page.locator("#captcha-img")
result = client.solve_click(captcha.screenshot(), prompt)

if result.unmatched:
    raise RuntimeError(f"missed targets: {result.unmatched}")  # reload instead

box = captcha.bounding_box()
natural = captcha.evaluate("el => [el.naturalWidth, el.naturalHeight]")
for point in result.points:                     # coordinates relative to the image
    page.mouse.click(
        box["x"] + point.x * box["width"] / natural[0],
        box["y"] + point.y * box["height"] / natural[1],
    )
    page.wait_for_timeout(120)                  # small pause between clicks
```

- See `clients/python/examples/playwright_click.py` for the full script.
- **Map coordinates**: points are relative to the captcha image; scale them by
  the element's rendered size vs `naturalWidth`/`naturalHeight`.
- **Prompt formats** are parsed for you: bracketed targets
  (`【明,天,会】`, `“天”`, `「甲乙」`) win over instruction words, and bare
  text works too (`请依次点击明天会`, `click in order: A B C`).
- **`unmatched` is the abort signal** — the recognizer could not identify a
  target character, so reload the captcha rather than clicking a partial
  sequence.
- **No prompt available?** Call `solve_click(image)` without it and match
  `result.boxes` (with `ch`, `score` and `center`) yourself — useful when the
  site renders the prompt in canvas or an image.
- **Recommended setup**: the fine-tuned glyph detector and recognizer, with
  the prompt as a charset constraint:
  ```
  kaptchowmin --ocr --det --manifest model/models.json \
          --det-model ft-small-det --rec-model ft-small
  ```
  Detection F1@0.5 is **96.8%** on the synthetic click set (bundled
  `stock-det`: 60.6%); `charset_range` — pass the prompt characters in the
  request — now supports fine-tuned PP-OCRv6 (CTC) rec models as well as the
  official ones, constraining each single-glyph recognition to the targets.
- **Isolated glyphs are harder than full captcha lines** for the recognizers
  (a lowercase `l` can come back as `t`/empty while `B`/`8` confuse).
  `charset_range` fixes most of it; `unmatched` is the signal to reload the
  captcha. Verified with the recommended setup: the noisy `【B,l,R,T】` and
  line `【V,W,K,Q】` samples both match all targets in order.

## 4. Etiquette, retries and failure handling

- The client already retries connection errors / 5xx / 429 with backoff and
  honors `Retry-After`. Do **not** wrap it in an infinite loop: catch
  `ServerError` / `NetworkError`, count attempts, and fail the task.
- Run the server with `--rate-limit` when several workers share it and leave
  `--max-concurrency` at its default (CPU cores). The server queues inference
  instead of thrashing the CPU.
- If the server is reachable beyond localhost, start it with `--api-key <key>`
  and pass `api_key=` to the client.
- Log `model`, `confidence` and the attempt number; accuracy regressions become
  visible immediately.
- Only solve captchas on systems you are allowed to automate, and follow the
  target site's terms and `robots.txt`.

## 5. Scrapy and other non-browser stacks

If there is no browser (plain `requests` / Scrapy), the flow is the same minus
the DOM: fetch the captcha image bytes and call the client.

```python
from captchowmin import Client

client = Client("http://127.0.0.1:8000")

def parse(self, response):
    image_url = response.css("#captcha-img::attr(src)").get()
    image = fetch(image_url)                      # your existing fetch helper
    result = client.solve(image)
    yield FormRequest.from_response(
        response,
        formdata={"username": "...", "captcha": result.text},
    )
```

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ServerError` HTTP 500 | the bytes you sent are not a valid image | screenshot the `<img>` element, not the page |
| `ServerError` HTTP 500 with an oversized screenshot | full-page PNG | screenshot just the captcha element |
| `NetworkError` | server not running / wrong `base_url` | `client.healthz()` |
| empty `text` | image not finished loading | wait for visible + `networkidle` |
| transparent-black PNG reads as empty | alpha ignored | `client.solve(image, png_fix=True)` |
| text consistently wrong | model mismatch or case expectations | try `--profile accurate` / `--rec-model stock-common`, check the site's case handling |
| slider always a few pixels short/long | calibration | adjust `offset`, verify `handle_x` includes the handle's centre |
| everything slow | `--profile accurate` on CPU | use `balanced` / `fast`, or a bigger `--max-concurrency` |
