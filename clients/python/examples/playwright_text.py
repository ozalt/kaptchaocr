"""Solve a text / math captcha and submit it with Playwright (sync API).

Prerequisites:
    pip install playwright && playwright install chromium
    # captcha server with the fine-tuned model:
    # ./kaptchaocr --ocr --manifest model/models.json --profile balanced

Adapt the selectors and the success condition to the target site, and keep
requests polite: one attempt per fresh captcha image, no hammering.
"""

from playwright.sync_api import sync_playwright

from kaptchaocr import Client, ServerError

URL = "https://example.com/login"
CAPTCHA_IMAGE = "#captcha-img"
CAPTCHA_INPUT = "#captcha"
SUBMIT = "#submit"
MAX_ATTEMPTS = 3


def main() -> None:
    client = Client("http://127.0.0.1:8000", timeout=10, retries=2)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(URL)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Screenshot only the captcha element; the bytes go straight to /solve.
            image = page.locator(CAPTCHA_IMAGE).screenshot()

            try:
                result = client.solve(image)
            except ServerError as exc:
                print(f"attempt {attempt}: server error: {exc}")
                page.reload()
                continue

            answer = result.math.answer if result.math else result.text
            print(
                f"attempt {attempt}: text={result.text!r} "
                f"answer={answer!r} model={result.model}"
            )

            page.fill(CAPTCHA_INPUT, answer)
            page.click(SUBMIT)
            page.wait_for_load_state("networkidle")

            if "/login" not in page.url:
                print("logged in")
                break

            # The captcha was wrong: reload to get a fresh image and retry.
            page.reload()
        else:
            raise RuntimeError(f"captcha not solved after {MAX_ATTEMPTS} attempts")

        browser.close()


if __name__ == "__main__":
    main()
