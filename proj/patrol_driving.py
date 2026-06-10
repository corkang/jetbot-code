import argparse
import signal
import sys
import time
from collections import deque

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
        device="cuda",
        dry_run=False,
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


class DrivingPatrol:
    """Low-rate driving loop for a two-label forward/turn_left CNN."""

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

    def setup(self):
        self.device = self._select_device(self.config.device)
        model_path = ensure_model_path(self.config.model_path)
        self.model, model_metadata = load_driving_model(model_path, self.device, labels=self.labels)
        self.labels = model_metadata.get("labels", self.labels)
        self.preprocessing = model_metadata.get("preprocessing", self.preprocessing)
        print("Loaded model with labels={} preprocessing={}.".format(self.labels, self.preprocessing))

        if self.config.dry_run:
            print("Dry run enabled. Model loaded; camera and robot are not started.")
            return

        from jetbot import Camera, Robot

        self.robot = Robot()
        self.camera = Camera.instance(width=self.config.camera_width, height=self.config.camera_height)
        time.sleep(1.0)
        print("Robot and camera initialized at {}x{}.".format(self.config.camera_width, self.config.camera_height))

    def _select_device(self, requested):
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

    def request_stop(self, *_args):
        self.stop_requested = True

    def run(self):
        self.setup()
        if self.config.dry_run:
            return

        print("Starting driving patrol. Press Ctrl+C to stop.")
        try:
            while not self.stop_requested:
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

                self._apply_action(action, reason, confidence, probs)
                time.sleep(self.config.inference_interval)
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            self.stop()

    def _smoothed_label(self):
        if not self.history:
            return "turn_left"
        forward_count = sum(1 for label in self.history if label == "forward")
        turn_count = len(self.history) - forward_count
        return "forward" if forward_count > turn_count else "turn_left"

    def _apply_action(self, action, reason, confidence, probs):
        print(
            "action={} reason={} confidence={:.3f} probs={}".format(
                action,
                reason,
                confidence,
                ["{:.2f}".format(p) for p in probs],
            )
        )
        if action == "forward":
            self._forward()
        else:
            self._turn_left(reason)

    def _forward(self):
        self.robot.set_motors(self.config.speed, self.config.speed)

    def _turn_left(self, reason):
        print("turn_left: {}".format(reason))
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
                print("camera stop skipped:", exc)
        print("Patrol stopped.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="JetBot two-label CNN driving patrol.")
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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Load model only; do not open JetBot camera or motors.")
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
    )
    patrol = DrivingPatrol(config)
    signal.signal(signal.SIGTERM, patrol.request_stop)
    signal.signal(signal.SIGINT, patrol.request_stop)
    patrol.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
