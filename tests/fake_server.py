"""A scripted OpenAI-compatible chat server for tests: no network, no model.

    with FakeServer() as srv:
        srv.script = [reply("hello")]            # one response per call, in order; the last repeats
        client = Client(store, base_url=srv.url)

Each script entry is a dict: {"text": ..., "usage": {...}, "status": 200, "reject": "max_tokens"}.
`reject` makes the server answer 400 mentioning that parameter when the request carries it
(hosted-shape probing); `status` 401 answers a denial; "empty" text answers an empty completion.
Every request body is kept in srv.requests.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply(text: str, prompt_tokens: int = 10, completion_tokens: int = 5, **kw) -> dict:
    return {"text": text, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, "status": 200, **kw}


class FakeServer:
    def __init__(self):
        self.script: list[dict] = [reply("ok")]
        self.requests: list[dict] = []
        self.calls = 0
        self._lock = threading.Lock()
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                with srv._lock:
                    i = min(srv.calls, len(srv.script) - 1)
                    entry = srv.script[i]
                    srv.calls += 1
                    srv.requests.append(body)
                if entry.get("reject") and entry["reject"] in body:
                    self._send(400, {"error": {"message": f"Unsupported parameter: '{entry['reject']}' is not supported with this model. Use 'max_completion_tokens' instead."}})
                    return
                if entry.get("reject_temperature") and "temperature" in body:
                    self._send(400, {"error": {"message": "temperature does not support 0.0 with this model. Only the default (1) value is supported."}})
                    return
                if entry.get("status", 200) != 200:
                    self._send(entry["status"], {"error": {"message": entry.get("message", "Incorrect API key provided")}})
                    return
                text = entry.get("text", "")
                usage = entry.get("usage")
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for k in range(0, len(text), 4):
                        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "delta": {"content": text[k:k + 4]}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    last = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
                    self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
                    return
                self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": entry.get("finish", "stop")}],
                                 "usage": usage})

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
