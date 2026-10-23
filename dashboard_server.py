"""
dashboard_server.py
----------------------
Lightweight local web dashboard, launched automatically from main.py.
Runs a background HTTP server (Python's built-in http.server — no new
dependency to install) that serves:

    GET /            -> the dashboard page (dashboard/dashboard.html)
    GET /api/state   -> JSON snapshot of the latest cycle + rolling history

main.py calls start() once at startup (opens your browser automatically)
and update_state(...) once per live_loop() cycle. The HTTP server and
live_loop run on different threads, so all shared state access goes
through a lock.
"""

from __future__ import annotations
import json
import threading
import webbrowser
import datetime as dt
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_lock = threading.Lock()
_state = {
    "status": "starting",   # "starting" until the first real cycle completes
    "last_updated": None,
    "history": {"timestamps": [], "atm_iv": [], "garch_forecast_vol": [],
                "total_gex": [], "spot": []},
}
_MAX_HISTORY_POINTS = 200   # caps memory/page size over a multi-hour session
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard" / "dashboard.html"


def update_state(snapshot: dict) -> None:
    """
    Called once per live_loop() cycle with the full metrics dict
    (scalars like spot/atm_iv/total_gex, plus 'top_oi_calls',
    'top_oi_puts', 'gamma_by_strike' lists, plus 'signal' sub-dict).
    Merges it into the shared state the dashboard polls, and appends
    the chart-relevant scalars to rolling history arrays.
    """
    with _lock:
        _state["status"] = "live"
        _state["last_updated"] = dt.datetime.now().strftime("%H:%M:%S")
        _state.update(snapshot)

        hist = _state["history"]
        hist["timestamps"].append(_state["last_updated"])
        hist["atm_iv"].append(snapshot.get("atm_iv"))
        hist["garch_forecast_vol"].append(snapshot.get("garch_forecast_vol"))
        hist["total_gex"].append(snapshot.get("total_gex"))
        hist["spot"].append(snapshot.get("spot"))
        for key in list(hist.keys()):
            hist[key] = hist[key][-_MAX_HISTORY_POINTS:]


def set_status_message(message: str) -> None:
    """Used before the first cycle completes (e.g. 'fitting volatility model...')
    so the page shows real progress instead of a blank screen."""
    with _lock:
        _state["status_message"] = message


def _json_default(obj):
    """
    Several values that flow into _state come from pandas/numpy
    operations (e.g. gamma_context's idxmax()/sums, chain.to_dict()'s
    numeric columns) and arrive as numpy scalar types, not plain
    Python ones — json.dumps() rejects those by default. This
    converts them rather than letting the whole state dump crash on
    the first cycle that happens to include one.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _get_state_json() -> bytes:
    with _lock:
        return json.dumps(_state, default=_json_default).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence per-request console noise; main.py's own prints are enough

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(_DASHBOARD_HTML_PATH, "text/html")
        elif self.path == "/api/state":
            body = _get_state_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path: Path, content_type: str):
        try:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"Dashboard file not found: {path}".encode())


def start(port: int = 8765, open_browser: bool = True) -> None:
    """
    Starts the dashboard's HTTP server on a background daemon thread
    (daemon=True so it never blocks main.py from exiting cleanly) and
    opens it in your default browser. Call once, near the top of main().
    """
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    print(f"[dashboard] running at {url}")
    if open_browser:
        webbrowser.open(url)
