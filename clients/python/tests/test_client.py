"""Offline tests for the kaptchaocr client (fake transport, no server)."""

import base64
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

from kaptchaocr import Client, NetworkError, ServerError
from kaptchaocr.client import TransportResponse


class FakeTransport:
    """Scripted transport: each call pops the next entry.

    An entry is either an exception (raised), or
    ``(status, headers, payload)`` / ``payload`` where ``payload`` may be a
    full envelope dict, a data dict (wrapped automatically) or raw bytes.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            SimpleNamespace(
                method=method,
                url=url,
                headers=dict(headers),
                body=body,
                timeout=timeout,
            )
        )

        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")

        item = self.responses.pop(0)

        if isinstance(item, Exception):
            raise item

        if isinstance(item, tuple):
            status, headers, payload = item
        else:
            status, headers, payload = 200, {}, item

        return TransportResponse(status=status, headers=headers, body=_to_body(payload))


def _to_body(payload):
    if isinstance(payload, bytes):
        return payload

    if isinstance(payload, str):
        return payload.encode("utf-8")

    if isinstance(payload, dict) and "code" not in payload:
        payload = {"code": 200, "msg": "success", "data": payload}

    return json.dumps(payload).encode("utf-8")


def make_client(*responses, **kwargs):
    transport = FakeTransport(*responses)
    # Trailing slash checks base_url normalization.
    client = Client("http://server:8000/", transport=transport, **kwargs)
    return client, transport


class SolveTest(unittest.TestCase):
    def test_solve_math(self):
        client, transport = make_client(
            {
                "text": "5+?=8",
                "math": {"expression": "5+?=8", "result": 3.0, "answer": "3"},
                "probability": None,
                "charset": None,
                "confidence": None,
                "model": "ft-small",
            }
        )

        result = client.solve(b"\x89PNG")

        self.assertEqual(result.text, "5+?=8")
        self.assertIsNotNone(result.math)
        self.assertEqual(result.math.answer, "3")
        self.assertEqual(result.math.result, 3.0)
        self.assertEqual(result.model, "ft-small")

        call = transport.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, "http://server:8000/solve")
        payload = json.loads(call.body)
        self.assertEqual(payload["image"], base64.b64encode(b"\x89PNG").decode("ascii"))
        self.assertEqual(payload["math"], "auto")

    def test_solve_without_math(self):
        client, transport = make_client({"text": "AB12", "math": None})

        result = client.solve(b"a", math="off", probability=True)

        self.assertIsNone(result.math)
        payload = json.loads(transport.calls[0].body)
        self.assertEqual(payload["math"], "off")
        self.assertTrue(payload["probability"])
        self.assertNotIn("charset_range", payload, "None fields must be omitted")

    def test_solve_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "captcha.png")

            with open(path, "wb") as handle:
                handle.write(b"file-bytes")

            client, transport = make_client({"text": "K9"})
            result = client.solve_file(path)

        self.assertEqual(result.text, "K9")
        payload = json.loads(transport.calls[0].body)
        self.assertEqual(payload["image"], base64.b64encode(b"file-bytes").decode("ascii"))

    def test_image_type_error(self):
        client, transport = make_client()

        with self.assertRaises(TypeError):
            client.solve(12345)

        self.assertEqual(transport.calls, [])

    def test_api_key_header(self):
        client, transport = make_client({"text": "x"}, api_key="secret")

        client.solve(b"a")

        self.assertEqual(transport.calls[0].headers["x-api-key"], "secret")


class SlideTest(unittest.TestCase):
    def test_slide_solve(self):
        client, transport = make_client(
            {
                "target": [140, 69, 200, 124],
                "target_x": 0,
                "target_y": 0,
                "gap_x": 140,
                "gap_y": 69,
                "distance": 120.0,
                "duration_ms": 482.0,
                "trajectory": [
                    {"x": 0.0, "y": 0.0, "t": 0.0},
                    {"x": 60.5, "y": -0.4, "t": 241.0},
                    {"x": 120.0, "y": 0.0, "t": 482.0},
                ],
            }
        )

        result = client.slide_solve(b"t", b"b", handle_x=20, handle_y=40, seed=7, offset=-2)

        self.assertEqual(result.gap_x, 140)
        self.assertEqual(result.distance, 120.0)
        self.assertEqual(len(result.trajectory), 3)
        self.assertEqual(result.trajectory[-1].x, 120.0)

        payload = json.loads(transport.calls[0].body)
        self.assertEqual(payload["handle_x"], 20)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["offset"], -2)
        self.assertNotIn("trajectory", payload)

    def test_slide_match_and_comparison(self):
        client, transport = make_client(
            {"target": [1, 2, 3, 4], "target_x": 0, "target_y": 5},
            {"x": 144, "y": 76},
        )

        matched = client.slide_match(b"t", b"b", simple_target=True)
        compared = client.slide_comparison(b"c", b"d")

        self.assertEqual(matched.target_y, 5)
        self.assertEqual(compared.x, 144)
        self.assertTrue(json.loads(transport.calls[0].body)["simple_target"])


class OtherRoutesTest(unittest.TestCase):
    def test_ocr_and_det(self):
        client, transport = make_client(
            {"text": "AB12", "probability": [[0.1]], "charset": ["A"], "confidence": 0.9},
            {"bboxes": [[1, 2, 3, 4]], "scores": [0.87]},
        )

        ocr = client.ocr(b"a")
        det = client.det(b"a")

        self.assertEqual(ocr.text, "AB12")
        self.assertEqual(ocr.confidence, 0.9)
        self.assertEqual(det.bboxes, [[1, 2, 3, 4]])
        self.assertEqual(det.scores, [0.87])

    def test_solve_click_with_prompt(self):
        client, transport = make_client(
            {
                "prompt": "VWKQ",
                "points": [{"x": 151.0, "y": 38.0}, {"x": 114.0, "y": 38.0}],
                "boxes": [
                    {"x1": 28, "y1": 22, "x2": 59, "y2": 61, "ch": "Q", "score": 0.34},
                    {"x1": 139, "y1": 23, "x2": 163, "y2": 53, "ch": "v", "score": 0.81},
                ],
                "unmatched": [],
            }
        )

        result = client.solve_click(b"img", "请依次点击【V,W】", score_threshold=0.4)

        self.assertEqual(result.prompt, "VWKQ")
        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.points[0].x, 151.0)
        self.assertEqual(result.boxes[0].ch, "Q")
        self.assertEqual(result.boxes[0].center, (43.5, 41.5))

        payload = json.loads(transport.calls[0].body)
        self.assertEqual(payload["prompt"], "请依次点击【V,W】")
        self.assertEqual(payload["score_threshold"], 0.4)
        self.assertNotIn("charset_range", payload)

    def test_solve_click_without_prompt(self):
        client, transport = make_client(
            {"prompt": None, "points": [], "boxes": [], "unmatched": ["明"]}
        )

        result = client.solve_click(b"img")

        self.assertIsNone(result.prompt)
        self.assertEqual(result.points, [])
        self.assertEqual(result.unmatched, ["明"])

        payload = json.loads(transport.calls[0].body)
        self.assertNotIn("prompt", payload)

    def test_status(self):
        client, transport = make_client(
            {"service_status": "running", "enabled_features": ["ocr", "slide"]}
        )

        status = client.status()

        self.assertEqual(status.service_status, "running")
        self.assertIn("slide", status.enabled_features)
        self.assertEqual(transport.calls[0].method, "GET")
        self.assertEqual(transport.calls[0].url, "http://server:8000/status")


class ErrorTest(unittest.TestCase):
    def test_4xx_raises_without_retry(self):
        client, transport = make_client((400, {}, {"code": 400, "msg": "bad request"}))

        with self.assertRaises(ServerError) as context:
            client.solve(b"a")

        self.assertEqual(context.exception.status, 400)
        self.assertEqual(context.exception.msg, "bad request")
        self.assertEqual(len(transport.calls), 1)

    def test_retries_5xx_then_succeeds(self):
        client, transport = make_client(
            (500, {}, {"code": 500, "msg": "boom"}),
            {"text": "ok"},
        )

        with mock.patch("kaptchaocr.client.time.sleep") as sleep:
            result = client.solve(b"a")

        self.assertEqual(result.text, "ok")
        self.assertEqual(len(transport.calls), 2)
        sleep.assert_called_once_with(0.2)

    def test_retry_after_is_respected(self):
        client, transport = make_client(
            (429, {"retry-after": "1.5"}, {"code": 429, "msg": "too many requests"}),
            {"text": "ok"},
        )

        with mock.patch("kaptchaocr.client.time.sleep") as sleep:
            client.solve(b"a")

        sleep.assert_called_once_with(1.5)
        self.assertEqual(len(transport.calls), 2)

    def test_network_error_after_retries(self):
        client, transport = make_client(URLError("down"), URLError("down"), URLError("down"))

        with mock.patch("kaptchaocr.client.time.sleep"):
            with self.assertRaises(NetworkError):
                client.solve(b"a")

        self.assertEqual(len(transport.calls), 3, "initial attempt + 2 retries")

    def test_invalid_json_response(self):
        client, _ = make_client((200, {}, b"<html>not json</html>"))

        with self.assertRaises(ServerError):
            client.solve(b"a")

    def test_healthz(self):
        client, _ = make_client((200, {}, b"ok"))
        self.assertTrue(client.healthz())

        client, _ = make_client(URLError("down"))
        self.assertFalse(client.healthz())


if __name__ == "__main__":
    unittest.main()
