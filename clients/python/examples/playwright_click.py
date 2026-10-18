"""Solve a click-text captcha with Playwright + /solve/click.

Prerequisites:
    pip install playwright && playwright install chromium
    # captcha server with both features:
    # ./kaptchowmin --ocr --det --manifest model/models.json --profile balanced

Flow: read the prompt text from the page, solve the captcha image, map the
returned image coordinates onto the (possibly scaled) element and click.
"""

from playwright.sync_api import sync_playwright

from captchowmin import Client

URL = "https://example.com/login"
CAPTCHA_IMAGE = "#captcha-img"     # the image the points are relative to
CAPTCHA_PROMPT = "#captcha-tip"    # element containing e.g. 请依次点击【明,天,会】
SUBMIT = "#submit"


def main() -> None:
    client = Client("http://127.0.0.1:8000", timeout=10)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(URL)

        captcha = page.locator(CAPTCHA_IMAGE)
        captcha.wait_for(state="visible")

        prompt = page.locator(CAPTCHA_PROMPT).inner_text()
        result = client.solve_click(captcha.screenshot(), prompt)

        print("prompt:", result.prompt, "| unmatched:", result.unmatched)
        if result.unmatched:
            raise RuntimeError(f"recognizer missed targets: {result.unmatched}")

        # Points are in captcha-image coordinates: map them onto the element
        # (the rendered size may differ from the natural image size).
        box = captcha.bounding_box()
        assert box is not None
        natural = captcha.evaluate("el => [el.naturalWidth, el.naturalHeight]")

        for point in result.points:
            page.mouse.click(
                box["x"] + point.x * box["width"] / natural[0],
                box["y"] + point.y * box["height"] / natural[1],
            )
            page.wait_for_timeout(120)  # small pause between clicks

        page.click(SUBMIT)
        page.wait_for_timeout(2000)
        print("submitted")

        browser.close()


if __name__ == "__main__":
    main()
