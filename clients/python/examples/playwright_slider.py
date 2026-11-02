"""Solve a slider captcha with Playwright, replaying the server trajectory.

Prerequisites:
    pip install playwright && playwright install chromium
    # captcha server started with --slide:
    # ./kaptchaocr --slide --manifest model/models.json --profile balanced

The server returns trajectory points as deltas from the handle start, sampled
at ~60 Hz; replay them with small waits so the movement is time-consistent.
"""

from playwright.sync_api import sync_playwright

from kaptchaocr import Client

URL = "https://example.com/login"
HANDLE = ".slider-handle"      # the draggable piece
TARGET = ".slider-target"      # small puzzle piece image (usually a <canvas>/<img>)
BACKGROUND = ".slider-bg"      # the background image
FRAME_MS = 16                  # ~60 Hz, matching the server sampling


def main() -> None:
    client = Client("http://127.0.0.1:8000", timeout=10)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(URL)

        handle = page.locator(HANDLE).bounding_box()
        assert handle is not None, "slider handle not found"

        handle_x = handle["x"] + handle["width"] / 2
        handle_y = handle["y"] + handle["height"] / 2

        result = client.slide_solve(
            page.locator(TARGET).screenshot(),
            page.locator(BACKGROUND).screenshot(),
            handle_x=handle_x,
            handle_y=handle_y,
            offset=-3,  # calibrate for sites whose gap is slightly offset
            seed=7,     # omit for a random tremor each attempt
        )
        print(
            f"gap_x={result.gap_x} distance={result.distance:.0f} "
            f"duration={result.duration_ms:.0f}ms points={len(result.trajectory)}"
        )

        page.mouse.move(handle_x, handle_y)
        page.mouse.down()

        for point in result.trajectory:
            page.mouse.move(handle_x + point.x, handle_y + point.y)
            page.wait_for_timeout(FRAME_MS)

        page.mouse.up()
        page.wait_for_timeout(2000)  # let the site verify the drag

        browser.close()


if __name__ == "__main__":
    main()
