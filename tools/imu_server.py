"""
imu_server.py — HTTP server nhận batch IMU từ watch.
Tự động mở file CSV khi session_id thay đổi.
"""
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class ImuServer(threading.Thread):
    def __init__(self, port: int, out_root: Path):
        super().__init__(daemon=True)
        self.port = port
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.current_session = None
        self.csv_file = None
        self.csv_writer = None
        self._lock = threading.Lock()
        self.total_samples = 0
        self.total_batches = 0
        self._last_log_sample = 0

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/imu_batch":
                    self.send_response(404); self.end_headers(); return
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                except Exception as e:
                    print(f"[IMU] JSON error: {e}")
                    self.send_response(400); self.end_headers(); return
                outer._write_batch(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, *a, **k):
                pass

        self.httpd = HTTPServer(("0.0.0.0", port), Handler)

    def run(self):
        print(f"[IMU Server] Lắng nghe trên 0.0.0.0:{self.port}")
        self.httpd.serve_forever()

    def _write_batch(self, data: dict):
        with self._lock:
            session_id = data["session_id"]
            if session_id != self.current_session:
                self._open_session(session_id)

            utc_start = int(data["utc_start_ms"])
            millis_at_start = int(data["millis_at_start"])

            for s in data["samples"]:
                sample_utc_ms = utc_start + (int(s["t_ms"]) - millis_at_start)
                self.csv_writer.writerow([
                    sample_utc_ms, s["t_ms"],
                    s["ax"], s["ay"], s["az"],
                    s["gx"], s["gy"], s["gz"],
                    s["bpm"], s["finger"],
                ])
                self.total_samples += 1
            self.total_batches += 1
            self.csv_file.flush()

            # Log progress mỗi 100 samples
            if self.total_samples - self._last_log_sample >= 100:
                self._last_log_sample = self.total_samples
                rate = self.total_samples / max(1, self.total_batches)
                print(f"[IMU] {self.total_samples} samples | "
                      f"{self.total_batches} batches | "
                      f"avg {rate:.1f}/batch")

    def _open_session(self, session_id: str):
        if self.csv_file:
            self.csv_file.close()
        self.current_session = session_id
        session_dir = self.out_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{session_id}_imu.csv"
        self.csv_file = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_utc_ms", "t_ms",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "bpm", "finger",
        ])
        print(f"[IMU] Mở file → {path}")

    def close(self):
        with self._lock:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.current_session = None