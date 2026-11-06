"""Payment and fulfilment provider, run in its own container beside the service under test.

The ledger is the verifier's source of truth. Its only control channel is this process's
stdin and stdout, which the host holds, so nothing on the sandbox network can read or change it.
Standard library only.
"""

import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = 8100
HOLD_LIMIT_SECONDS = 30.0


class Ledger:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.charges: dict[str, dict[str, object]] = {}
        self.shipments: list[str] = []
        self.charge_calls = 0
        self.transient_failures: dict[str, int] = {}
        self.latency_seconds = 0.0
        self.released = threading.Event()
        self.released.set()
        self.entry_open = threading.Event()
        self.entry_open.set()
        self.slow_once: dict[str, float] = {}

    def charge(self, body: dict[str, object]) -> int:
        order_id = str(body.get("order_id"))
        with self.lock:
            self.charge_calls += 1
        if not self.entry_open.is_set():
            # This call dies with its worker: it never reaches the payment network.
            self.entry_open.wait(HOLD_LIMIT_SECONDS)
            return 503
        with self.lock:
            remaining = self.transient_failures.get(order_id, 0)
            if remaining > 0:
                self.transient_failures[order_id] = remaining - 1
                return 503
            if body.get("currency") == "XXX":
                return 422
            self.charges.setdefault(
                str(body.get("key")),
                {
                    "order_id": order_id,
                    "amount_cents": body.get("amount_cents"),
                    "currency": body.get("currency"),
                },
            )
        self.released.wait(HOLD_LIMIT_SECONDS)
        with self.lock:
            delay = self.slow_once.pop(order_id, self.latency_seconds)
        time.sleep(delay)
        return 200

    def ship(self, body: dict[str, object]) -> int:
        with self.lock:
            self.shipments.append(str(body.get("order_id")))
        return 200

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "charges": dict(self.charges),
                "shipments": list(self.shipments),
                "charge_calls": self.charge_calls,
            }


LEDGER = Ledger()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self.respond(400)
            return
        if not isinstance(body, dict):
            self.respond(400)
        elif self.path == "/charges":
            self.respond(LEDGER.charge(body))
        elif self.path == "/shipments":
            self.respond(LEDGER.ship(body))
        else:
            self.respond(404)

    def respond(self, status: int) -> None:
        payload = json.dumps({"status": status}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


def forward(command: dict[str, Any]) -> dict[str, Any]:
    """Send a client request to the service, which the host cannot reach directly."""
    body = command.get("json")
    request = urllib.request.Request(
        str(command["url"]),
        method=str(command.get("method", "GET")),
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=float(command["request_timeout"])) as response:
            status, text = response.status, response.read(65_536).decode(errors="replace")
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read(65_536).decode(errors="replace")
    except (OSError, http.client.HTTPException, ValueError) as error:
        return {"connection_error": repr(error), "seconds": time.perf_counter() - started}
    return {"status": status, "text": text, "seconds": time.perf_counter() - started}


def execute(command: dict[str, Any]) -> dict[str, Any]:
    operation = command["op"]
    if operation == "request":
        return forward(command)
    if operation == "ledger":
        return LEDGER.snapshot()
    if operation == "load":
        with LEDGER.lock:
            LEDGER.charges = dict(command["charges"])
            LEDGER.shipments = list(command["shipments"])
            LEDGER.charge_calls = int(command["charge_calls"])
        return {}
    if operation == "fail_charges":
        with LEDGER.lock:
            LEDGER.transient_failures[str(command["order_id"])] = int(command["times"])
        return {}
    if operation == "latency":
        LEDGER.latency_seconds = float(command["seconds"])
        return {}
    if operation == "close_entry":
        LEDGER.entry_open.clear()
        return {}
    if operation == "open_entry":
        LEDGER.entry_open.set()
        return {}
    if operation == "slow_once":
        with LEDGER.lock:
            LEDGER.slow_once[str(command["order_id"])] = float(command["seconds"])
        return {}
    if operation == "hold":
        LEDGER.released.clear()
        return {}
    if operation == "release":
        LEDGER.released.set()
        return {}
    raise ValueError(f"unknown operation: {operation}")


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            print(json.dumps(execute(json.loads(line))), flush=True)
        except Exception as error:
            print(json.dumps({"error": repr(error)}), flush=True)


if __name__ == "__main__":
    main()
