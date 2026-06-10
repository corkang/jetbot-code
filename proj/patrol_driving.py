import argparse
import json
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import torch

from driving_model import DRIVING_LABELS, ensure_model_path, load_driving_model, predict_label


class PatrolConfig:
    def __init__(
        self,
        model_path,
        image_size=224,
        speed=0.4,
        turn_speed=0.4,
        turn_seconds=0.16,
        inference_interval=0.25,
        confidence_threshold=0.55,
        smooth_window=3,
        camera_width=224,
        camera_height=224,
        device="cpu",
        dry_run=False,
        patrol_time=None,
        capture_interval=1.0,
        capture_dir="img/patrol",
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.speed = speed
        self.turn_speed = turn_speed
        self.turn_seconds = turn_seconds
        self.inference_interval = inference_interval
        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.device = device
        self.dry_run = dry_run
        self.patrol_time = patrol_time
        self.capture_interval = capture_interval
        self.capture_dir = capture_dir


class DrivingPatrol:
    """Safety-CNN-only patrol with timed image capture for offline detection."""

    def __init__(self, config):
        self.config = config
        self.labels = list(DRIVING_LABELS)
        self.stop_requested = False
        self.history = deque(maxlen=max(1, config.smooth_window))
        self.robot = None
        self.camera = None
        self.model = None
        self.preprocessing = "raw_rgb_normalized"
        self.device = torch.device("cpu")
        self.capture_run_dir = None
        self.capture_records = []
        self.capture_count = 0
        self.last_capture_time = 0.0

    def setup(self):
        print("Patrol setup started.", flush=True)
        self.device = self._select_device(self.config.device)
        print("Using device={}.".format(self.device), flush=True)

        model_path = ensure_model_path(self.config.model_path)
        print("Model path OK: {}".format(model_path), flush=True)
        print("Loading model checkpoint...", flush=True)
        self.model, model_metadata = load_driving_model(model_path, self.device, labels=self.labels)
        self.labels = model_metadata.get("labels", self.labels)
        self.preprocessing = model_metadata.get("preprocessing", self.preprocessing)
        print("Loaded model with labels={} preprocessing={}.".format(self.labels, self.preprocessing), flush=True)

        if self.config.dry_run:
            print("Dry run enabled. Model loaded; camera and robot are not started.", flush=True)
            return

        from jetbot import Camera, Robot

        self.robot = Robot()
        self.camera = Camera.instance(width=self.config.camera_width, height=self.config.camera_height)
        time.sleep(1.0)
        print("Robot and camera initialized at {}x{}.".format(self.config.camera_width, self.config.camera_height), flush=True)
        if self.camera.value is not None:
            print("Camera frame shape: {}.".format(self.camera.value.shape), flush=True)

        self.capture_run_dir = self._make_capture_run_dir()
        print("Capture directory: {}".format(self.capture_run_dir), flush=True)

    def _select_device(self, requested):
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

    def _make_capture_run_dir(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.config.capture_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def request_stop(self, *_args):
        self.stop_requested = True

    def run(self):
        self.setup()
        if self.config.dry_run:
            return

        start_time = time.time()
        deadline = None
        if self.config.patrol_time is not None:
            deadline = start_time + max(0.0, float(self.config.patrol_time))
            print("Starting timed patrol for {:.1f}s. Press Ctrl+C to stop.".format(float(self.config.patrol_time)), flush=True)
        else:
            print("Starting open-ended patrol. Press Ctrl+C to stop.", flush=True)

        try:
            while not self.stop_requested:
                now = time.time()
                if deadline is not None and now >= deadline:
                    print("Patrol time reached.", flush=True)
                    break

                frame = self.camera.value
                if frame is None:
                    self._turn_left("no_frame")
                    time.sleep(self.config.inference_interval)
                    continue

                label, confidence, probs = predict_label(
                    self.model,
                    frame,
                    device=self.device,
                    labels=self.labels,
                    image_size=self.config.image_size,
                    preprocessing=self.preprocessing,
                )

                if confidence < self.config.confidence_threshold:
                    action = "turn_left"
                    reason = "low_confidence"
                elif label == "turn_left":
                    self.history.clear()
                    action = "turn_left"
                    reason = "model"
                else:
                    self.history.append(label)
                    action = self._smoothed_label()
                    reason = "model"

                self._maybe_capture_frame(
                    frame=frame,
                    elapsed=now - start_time,
                    action=action,
                    label=label,
                    confidence=confidence,
                    probs=probs,
                )
                self._apply_action(action, reason, confidence, probs)
                time.sleep(self.config.inference_interval)
        except KeyboardInterrupt:
            print("Interrupted.", flush=True)
        finally:
            self.stop()
            self._write_capture_metadata()

    def _smoothed_label(self):
        if not self.history:
            return "turn_left"
        forward_count = sum(1 for label in self.history if label == "forward")
        turn_count = len(self.history) - forward_count
        return "forward" if forward_count > turn_count else "turn_left"

    def _maybe_capture_frame(self, frame, elapsed, action, label, confidence, probs):
        if self.capture_run_dir is None:
            return

        now = time.time()
        if self.capture_count > 0 and now - self.last_capture_time < self.config.capture_interval:
            return

        self.capture_count += 1
        self.last_capture_time = now
        file_name = "{:04d}.jpg".format(self.capture_count)
        output_path = self.capture_run_dir / file_name
        cv2.imwrite(str(output_path), frame)

        record = {
            "file_name": file_name,
            "elapsed": float(elapsed),
            "action": action,
            "raw_label": label,
            "confidence": float(confidence),
            "probabilities": [float(prob) for prob in probs],
        }
        self.capture_records.append(record)
        print("captured {} elapsed={:.1f}s action={} confidence={:.3f}".format(file_name, elapsed, action, confidence), flush=True)

    def _write_capture_metadata(self):
        if self.capture_run_dir is None:
            return
        metadata_path = self.capture_run_dir / "metadata.json"
        metadata = {
            "capture_interval": self.config.capture_interval,
            "patrol_time": self.config.patrol_time,
            "camera_width": self.config.camera_width,
            "camera_height": self.config.camera_height,
            "records": self.capture_records,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print("Capture metadata: {}".format(metadata_path), flush=True)
        print("Captured frames: {}".format(len(self.capture_records)), flush=True)

    def _apply_action(self, action, reason, confidence, probs):
        print(
            "action={} reason={} confidence={:.3f} probs={}".format(
                action,
                reason,
                confidence,
                ["{:.2f}".format(p) for p in probs],
            ),
            flush=True,
        )
        if action == "forward":
            self._forward()
        else:
            self._turn_left(reason)

    def _forward(self):
        self.robot.set_motors(self.config.speed, self.config.speed)

    def _turn_left(self, reason):
        print("turn_left: {}".format(reason), flush=True)
        self.robot.set_motors(-self.config.turn_speed, self.config.turn_speed)
        time.sleep(self.config.turn_seconds)
        self.robot.stop()

    def stop(self):
        if self.robot is not None:
            self.robot.stop()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as exc:
                print("camera stop skipped:", exc, flush=True)
        print("Patrol stopped.", flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="JetBot safety-CNN patrol with timed image capture.")
    parser.add_argument("--model", required=True, help="Path to safety CNN checkpoint.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--turn-speed", type=float, default=0.4)
    parser.add_argument("--turn-seconds", type=float, default=0.16)
    parser.add_argument("--inference-interval", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Load model only; do not open JetBot camera or motors.")
    parser.add_argument("--patrol-time", type=float, default=None, help="Seconds to patrol before stopping.")
    parser.add_argument("--capture-interval", type=float, default=1.0, help="Seconds between saved patrol frames.")
    parser.add_argument("--capture-dir", default="img/patrol", help="Directory for timed patrol frame captures.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = PatrolConfig(
        model_path=args.model,
        image_size=args.image_size,
        speed=args.speed,
        turn_speed=args.turn_speed,
        turn_seconds=args.turn_seconds,
        inference_interval=args.inference_interval,
        confidence_threshold=args.confidence_threshold,
        smooth_window=args.smooth_window,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        device=args.device,
        dry_run=args.dry_run,
        patrol_time=args.patrol_time,
        capture_interval=args.capture_interval,
        capture_dir=args.capture_dir,
    )
    patrol = DrivingPatrol(config)
    signal.signal(signal.SIGTERM, patrol.request_stop)
    signal.signal(signal.SIGINT, patrol.request_stop)
    patrol.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
