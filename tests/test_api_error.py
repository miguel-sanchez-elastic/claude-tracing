"""API failures should print the provider message instead of a traceback."""

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CREDIT_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)


class RejectingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path != "/v1/messages":
            self.send_error(404)
            return
        payload = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": CREDIT_MESSAGE,
                },
                "request_id": "req_test_credit",
            }
        ).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("request-id", "req_test_credit")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class ApiErrorTest(unittest.TestCase):
    def test_credit_balance_error_is_printed_without_traceback(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), RejectingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = os.environ.copy()
            env.update(
                {
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                }
            )
            result = subprocess.run(
                [sys.executable, "-m", "claude_tracing", "--console", "hello"],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("Traceback", result.stdout)
        self.assertIn(f"Claude API error: {CREDIT_MESSAGE}", result.stderr)
        self.assertIn("Request ID: req_test_credit", result.stderr)
