"""The provider as agents see it while working: same behaviour, plus a read-only ledger view.

Scoring never uses this. It runs the real provider in a separate sandbox.
"""

import json
from http.server import ThreadingHTTPServer

from provider import LEDGER, PORT, Handler


class DevelopmentHandler(Handler):
    def do_GET(self) -> None:
        if self.path != "/ledger":
            self.respond(404)
            return
        payload = json.dumps(LEDGER.snapshot()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), DevelopmentHandler).serve_forever()
