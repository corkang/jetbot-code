import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import cv2
from jetbot import Camera
from servoserial import ServoSerial


CAMERA_SIZES = [(720, 720), (224, 224), (300, 300), (480, 480)]
IMAGE_SIZE = 480
IMG_ROOT = Path("img")
KEY_TO_LABEL = {
    "w": "forward",
    "a": "turn_left",
    "d": "turn_right",
    "s": "stop_reverse",
    "g": "object",
}
LABELS = list(KEY_TO_LABEL.values())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DatasetCollector:
    def __init__(self, img_root: Path = IMG_ROOT) -> None:
        self.img_root = img_root
        self.camera = None
        self.latest_jpeg = None  # type: Optional[bytes]
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None  # type: Optional[threading.Thread]

        for label in LABELS:
            (self.img_root / label).mkdir(parents=True, exist_ok=True)

    def set_servo_default(self) -> None:
        servo_device = ServoSerial()
        servo_device.Servo_serial_double_control(1, 1000, 2, 1200)
        time.sleep(0.5)

    def reset_camera_singleton(self) -> None:
        old_camera = getattr(Camera, "_instance", None)
        if old_camera is not None:
            try:
                old_camera.stop()
            except Exception as exc:
                print("Previous camera stop skipped:", exc)
        try:
            Camera.clear_instance()
        except Exception:
            Camera._instance = None

    def open_camera(self) -> None:
        errors = []
        for width, height in CAMERA_SIZES:
            self.reset_camera_singleton()
            time.sleep(0.5)
            try:
                camera = Camera.instance(width=width, height=height)
                if camera.value is None:
                    raise RuntimeError("Camera opened but first frame is None")
                self.camera = camera
                print(f"Camera initialized at {width}x{height}; stream/save size is {IMAGE_SIZE}x{IMAGE_SIZE}.")
                return
            except Exception as exc:
                errors.append(f"{width}x{height}: {exc}")
                print(f"Camera init failed at {width}x{height}: {exc}")
        raise RuntimeError("Could not initialize camera. Tried: " + " | ".join(errors))

    def frame_to_jpeg(self, frame) -> bytes:
        if frame.shape[0] != IMAGE_SIZE or frame.shape[1] != IMAGE_SIZE:
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode camera frame as JPEG")
        return encoded.tobytes()

    def start(self) -> None:
        self.set_servo_default()
        self.open_camera()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            frame = self.camera.value if self.camera is not None else None
            if frame is not None:
                try:
                    jpeg = self.frame_to_jpeg(frame)
                    with self.lock:
                        self.latest_jpeg = jpeg
                except Exception as exc:
                    print("Frame encode failed:", exc)
            time.sleep(0.03)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.camera is not None:
            self.camera.stop()

    def counts(self) -> Dict[str, int]:
        return {label: len(list((self.img_root / label).glob("*.jpg"))) for label in LABELS}

    def next_image_path(self, label: str) -> Path:
        ids = []
        for path in (self.img_root / label).glob("*.jpg"):
            if path.stem.isdigit():
                ids.append(int(path.stem))
        return self.img_root / label / f"{max(ids, default=0) + 1}.jpg"

    def capture(self, label: str) -> Path:
        if label not in LABELS:
            raise ValueError(f"Invalid label: {label}")
        with self.lock:
            jpeg = self.latest_jpeg
        if not jpeg:
            raise RuntimeError("No camera frame available yet")
        path = self.next_image_path(label)
        path.write_bytes(jpeg)
        return path


def make_handler(collector: DatasetCollector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict, status: int = 200) -> None:
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/counts":
                self.send_json({"counts": collector.counts()})
                return
            if parsed.path == "/capture":
                query = parse_qs(parsed.query)
                label = query.get("label", [""])[0]
                try:
                    path = collector.capture(label)
                    self.send_json({"ok": True, "label": label, "path": str(path), "counts": collector.counts()})
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if parsed.path == "/stream":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    with collector.lock:
                        jpeg = collector.latest_jpeg
                    if jpeg:
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    time.sleep(0.05)
                return
            self.send_json({"ok": False, "error": "Not found"}, status=404)

    return Handler


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>JetBot Dataset Collector</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
    img { width: 480px; height: 480px; border: 1px solid #444; background: #222; }
    button { margin: 4px; padding: 10px 14px; font-size: 16px; }
    .ok { color: #8f8; }
    .err { color: #f88; }
    code { color: #9cf; }
  </style>
</head>
<body>
  <h2>JetBot Dataset Collector</h2>
  <img src="/stream" alt="camera stream">
  <p>Keys: <code>w</code> forward, <code>a</code> turn_left, <code>d</code> turn_right, <code>s</code> stop_reverse, <code>g</code> object</p>
  <div id="buttons"></div>
  <pre id="counts"></pre>
  <p id="status"></p>
  <script>
    const keyToLabel = {w: "forward", a: "turn_left", d: "turn_right", s: "stop_reverse", g: "object"};
    const buttons = document.getElementById("buttons");
    const statusEl = document.getElementById("status");
    const countsEl = document.getElementById("counts");

    for (const [key, label] of Object.entries(keyToLabel)) {
      const button = document.createElement("button");
      button.textContent = `${key}: ${label}`;
      button.onclick = () => capture(label);
      buttons.appendChild(button);
    }

    async function refreshCounts() {
      const response = await fetch("/counts");
      const data = await response.json();
      countsEl.textContent = JSON.stringify(data.counts, null, 2);
    }

    async function capture(label) {
      const response = await fetch(`/capture?label=${encodeURIComponent(label)}`);
      const data = await response.json();
      if (data.ok) {
        statusEl.className = "ok";
        statusEl.textContent = `saved ${data.label}: ${data.path}`;
        countsEl.textContent = JSON.stringify(data.counts, null, 2);
      } else {
        statusEl.className = "err";
        statusEl.textContent = data.error;
      }
    }

    document.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      const label = keyToLabel[key];
      if (!label || event.repeat) return;
      event.preventDefault();
      capture(label);
    });

    refreshCounts();
    setInterval(refreshCounts, 3000);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="JetBot browser-based dataset collector.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    collector = DatasetCollector()
    collector.start()

    server = ThreadedHTTPServer((args.host, args.port), make_handler(collector))
    print(f"Open http://<jetbot-ip>:{args.port} from your MacBook.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping collector...")
    finally:
        server.shutdown()
        collector.stop()


if __name__ == "__main__":
    main()
