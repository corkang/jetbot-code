import argparse
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch

from detection_model import (
    DETECTION_STATES,
    detect_object,
    ensure_detection_model_path,
    load_detection_model,
)
from driving_model import DRIVING_LABELS, ensure_model_path, load_driving_model, predict_label
from object_reporter import LocalObjectReporter


class PatrolConfig:
    def __init__(
        self,
        model_path,
        image_size=224,
        speed=0.4,
        approach_speed=0.18,
        turn_speed=0.4,
        turn_seconds=0.16,
        inference_interval=0.25,
        confidence_threshold=0.55,
        smooth_window=3,
        camera_width=224,
        camera_height=224,
        device="cpu",
        dry_run=False,
        detection_model_path=None,
        detection_interval=0.75,
        detection_image_size=None,
        detection_result_ttl=8.0,
        require_fresh_detection_for_full_speed=True,
        detection_save_dir="img/detected",
        detection_save_cooldown=8.0,
        detection_absence_required=3,
        detection_center_deadband=0.16,
        detection_prob_threshold=0.50,
        detection_no_object_area_threshold=0.003,
        detection_close_area_threshold=0.080,
        detection_close_bbox_height_threshold=0.35,
        detection_on_close_only=False,
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.speed = speed
        self.approach_speed = approach_speed
        self.turn_speed = turn_speed
        self.turn_seconds = turn_seconds
        self.inference_interval = inference_interval
        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.device = device
        self.dry_run = dry_run
        self.detection_model_path = detection_model_path
        self.detection_interval = detection_interval
        self.detection_image_size = detection_image_size
        self.detection_result_ttl = detection_result_ttl
        self.require_fresh_detection_for_full_speed = require_fresh_detection_for_full_speed
        self.detection_save_dir = detection_save_dir
        self.detection_save_cooldown = detection_save_cooldown
        self.detection_absence_required = detection_absence_required
        self.detection_center_deadband = detection_center_deadband
        self.detection_prob_threshold = detection_prob_threshold
        self.detection_no_object_area_threshold = detection_no_object_area_threshold
        self.detection_close_area_threshold = detection_close_area_threshold
        self.detection_close_bbox_height_threshold = detection_close_bbox_height_threshold
        self.detection_on_close_only = detection_on_close_only


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
        self.detection_model = None
        self.detection_metadata = None
        self.reporter = None
        self.preprocessing = "raw_rgb_normalized"
        self.device = torch.device("cpu")
        self.last_detection_time = 0.0
        self.last_detection_save_time = 0.0
        self.detected_save_dir = Path(config.detection_save_dir)
        self.detection_stop = threading.Event()
        self.detection_lock = threading.Lock()
        self.detection_thread = None
        self.latest_detection_frame = None
        self.latest_detection_result = None
        self.detection_result_seq = 0
        self.last_logged_detection_seq = 0
        self.last_handled_detection_seq = 0
        self.report_suppressed_until_object_gone = False
        self.object_absence_count = 0
        self.object_approach_active = False
        self.object_approach_action = None

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
        self._load_detection_if_requested()

        if self.config.dry_run:
            print("Dry run enabled. Models loaded; camera and robot are not started.", flush=True)
            return

        from jetbot import Camera, Robot

        self.robot = Robot()
        self.camera = Camera.instance(width=self.config.camera_width, height=self.config.camera_height)
        time.sleep(1.0)
        print("Robot and camera initialized at {}x{}.".format(self.config.camera_width, self.config.camera_height), flush=True)
        if self.camera.value is not None:
            print("Camera frame shape: {}.".format(self.camera.value.shape), flush=True)
        if self.detection_model is not None:
            self.reporter = LocalObjectReporter(self.detected_save_dir)
            print("Detection saves enabled: {}.".format(self.detected_save_dir), flush=True)
            self._start_detection_worker()

    def _load_detection_if_requested(self):
        if not self.config.detection_model_path:
            print("Detection model not configured; running safety driving only.", flush=True)
            return

        detection_model_path = ensure_detection_model_path(self.config.detection_model_path)
        print("Detection model path OK: {}".format(detection_model_path), flush=True)
        print("Loading detection checkpoint...", flush=True)
        self.detection_model, self.detection_metadata = load_detection_model(
            detection_model_path,
            self.device,
        )
        print(
            "Loaded detection model with labels={} input={}x{}.".format(
                self.detection_metadata.get("class_names"),
                self.detection_metadata.get("image_width"),
                self.detection_metadata.get("image_height"),
            ),
            flush=True,
        )

    def _select_device(self, requested):
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

    def request_stop(self, *_args):
        self.stop_requested = True
        self.detection_stop.set()

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

                self._submit_detection_frame(frame, action)
                detection_action = self._maybe_plan_object_action(action)
                final_action = detection_action or action
                self._apply_action(final_action, reason, confidence, probs)
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
        elif action == "forward_slow":
            self._forward_slow()
        elif action == "turn_right":
            self._turn_right("object_center")
        elif action == "stop":
            self.robot.stop()
        else:
            self._turn_left(reason)

    def _start_detection_worker(self):
        if self.detection_thread is not None and self.detection_thread.is_alive():
            return
        self.detection_stop.clear()
        self.detection_thread = threading.Thread(target=self._detection_worker_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        print(
            "Detection worker started at interval={}s.".format(
                self.config.detection_interval
            ),
            flush=True,
        )

    def _submit_detection_frame(self, frame, safety_action):
        if self.detection_model is None:
            return
        if safety_action != "forward":
            self.object_approach_active = False
            with self.detection_lock:
                self.latest_detection_frame = None
            return
        with self.detection_lock:
            self.latest_detection_frame = frame.copy()

    def _detection_worker_loop(self):
        while not self.detection_stop.is_set():
            now = time.time()
            if now - self.last_detection_time < self.config.detection_interval:
                time.sleep(0.03)
                continue

            with self.detection_lock:
                frame = None if self.latest_detection_frame is None else self.latest_detection_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue

            self.last_detection_time = now
            start = time.time()
            metadata = self.detection_metadata or {}
            postprocess = metadata.get("postprocess_config", {})
            detection_image_size = self.config.detection_image_size
            if detection_image_size is None:
                image_width = int(metadata.get("image_width", self.config.camera_width))
                image_height = int(metadata.get("image_height", self.config.camera_height))
            else:
                image_width = int(detection_image_size)
                image_height = int(detection_image_size)
            try:
                summary, _mask = detect_object(
                    self.detection_model,
                    frame,
                    device=self.device,
                    image_width=image_width,
                    image_height=image_height,
                    prob_threshold=float(postprocess.get("inference_prob_threshold", self.config.detection_prob_threshold)),
                    no_object_area_threshold=float(postprocess.get("no_object_area_threshold", self.config.detection_no_object_area_threshold)),
                    close_area_threshold=float(postprocess.get("close_area_threshold", self.config.detection_close_area_threshold)),
                    close_bbox_height_threshold=float(postprocess.get("close_bbox_height_threshold", self.config.detection_close_bbox_height_threshold)),
                )
            except Exception as exc:
                print("detection worker error: {}".format(exc), flush=True)
                time.sleep(0.5)
                continue

            elapsed = time.time() - start
            with self.detection_lock:
                self.detection_result_seq += 1
                self.latest_detection_result = {
                    "seq": self.detection_result_seq,
                    "summary": summary,
                    "frame": frame,
                    "elapsed": elapsed,
                    "time": time.time(),
                }

    def _get_latest_detection_result(self):
        with self.detection_lock:
            if self.latest_detection_result is None:
                return None
            result = dict(self.latest_detection_result)
        return result

    def _maybe_plan_object_action(self, safety_action):
        if self.detection_model is None:
            return None
        if safety_action != "forward":
            self.object_approach_active = False
            return None

        now = time.time()
        result = self._get_latest_detection_result()
        if result is None:
            return self._stale_detection_action()

        seq = int(result["seq"])
        summary = result["summary"]
        result_age = now - float(result["time"])
        if result_age > self.config.detection_result_ttl:
            self.object_approach_active = False
            self.object_approach_action = None
            return self._stale_detection_action()

        if seq != self.last_logged_detection_seq:
            self.last_logged_detection_seq = seq
            print(
                "detection state={} area={:.3f} bbox={} max_prob={:.2f} elapsed={:.2f}s age={:.2f}s".format(
                    summary["state"],
                    summary["area_ratio"],
                    summary["bbox"],
                    summary["max_probability"],
                    result["elapsed"],
                    result_age,
                ),
                flush=True,
            )

        if self.report_suppressed_until_object_gone:
            if seq != self.last_handled_detection_seq:
                self.last_handled_detection_seq = seq
                self._handle_suppressed_detection(summary)
            return None

        if summary["state"] == DETECTION_STATES["none"]:
            self.object_approach_active = False
            self.object_approach_action = None
            return None

        if summary["state"] == DETECTION_STATES["close"]:
            if seq != self.last_handled_detection_seq:
                self.last_handled_detection_seq = seq
                self._report_detected_object(result["frame"], summary, now)
            return "stop"

        self.object_approach_active = True
        self.object_approach_action = self._object_approach_action(summary)
        return self.object_approach_action

    def _stale_detection_action(self):
        if self.config.require_fresh_detection_for_full_speed:
            return "forward_slow"
        return None

    def _handle_suppressed_detection(self, summary):
        self.object_approach_active = False
        self.object_approach_action = None
        if summary["state"] == DETECTION_STATES["none"]:
            self.object_absence_count += 1
            print(
                "report suppression: absence {}/{}".format(
                    self.object_absence_count,
                    self.config.detection_absence_required,
                ),
                flush=True,
            )
            if self.object_absence_count >= self.config.detection_absence_required:
                self.report_suppressed_until_object_gone = False
                self.object_absence_count = 0
                print("report suppression cleared; detection reporting re-armed.", flush=True)
        else:
            self.object_absence_count = 0
        return None

    def _object_approach_action(self, summary):
        center_x = summary.get("center_x_ratio")
        if center_x is None:
            return "forward_slow"
        half_deadband = self.config.detection_center_deadband / 2.0
        if center_x < 0.5 - half_deadband:
            return "turn_left"
        if center_x > 0.5 + half_deadband:
            return "turn_right"
        return "forward_slow"

    def _report_detected_object(self, frame, summary, now):
        if now - self.last_detection_save_time < self.config.detection_save_cooldown:
            self.report_suppressed_until_object_gone = True
            self.object_absence_count = 0
            return

        if self.robot is not None:
            self.robot.stop()
        if self.reporter is None:
            self.reporter = LocalObjectReporter(self.detected_save_dir)
        image_path, metadata_path = self.reporter.report(frame, summary)
        self.last_detection_save_time = now
        self.report_suppressed_until_object_gone = True
        self.object_absence_count = 0
        self.object_approach_active = False
        self.object_approach_action = None
        print(
            "Saved object report image={} metadata={}".format(
                image_path,
                metadata_path,
            ),
            flush=True,
        )

    def _forward(self):
        self.robot.set_motors(self.config.speed, self.config.speed)

    def _forward_slow(self):
        self.robot.set_motors(self.config.approach_speed, self.config.approach_speed)

    def _turn_left(self, reason):
        print("turn_left: {}".format(reason))
        self.robot.set_motors(-self.config.turn_speed, self.config.turn_speed)
        time.sleep(self.config.turn_seconds)
        self.robot.stop()

    def _turn_right(self, reason):
        print("turn_right: {}".format(reason))
        self.robot.set_motors(self.config.turn_speed, -self.config.turn_speed)
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
        self.detection_stop.set()
        if self.detection_thread is not None and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        print("Patrol stopped.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="JetBot two-label CNN driving patrol.")
    parser.add_argument("--model", required=True, help="Path to safety CNN checkpoint.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--approach-speed", type=float, default=0.18)
    parser.add_argument("--turn-speed", type=float, default=0.4)
    parser.add_argument("--turn-seconds", type=float, default=0.16)
    parser.add_argument("--inference-interval", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Load model only; do not open JetBot camera or motors.")
    parser.add_argument("--detection-model", default=None, help="Optional object segmentation checkpoint.")
    parser.add_argument("--detection-interval", type=float, default=0.75, help="Seconds between detection CNN runs.")
    parser.add_argument("--detection-image-size", type=int, default=None, help="Optional smaller square input size for detection, for example 160 or 128.")
    parser.add_argument("--detection-result-ttl", type=float, default=8.0, help="Seconds before a detection result is treated as stale.")
    parser.add_argument("--allow-full-speed-without-detection-result", action="store_true", help="Do not slow down while waiting for a fresh detection result.")
    parser.add_argument("--detection-save-dir", default="img/detected", help="Directory for detected object images.")
    parser.add_argument("--detection-save-cooldown", type=float, default=8.0, help="Minimum seconds between saved detection photos.")
    parser.add_argument("--detection-absence-required", type=int, default=3, help="No-object detections required before reporting is re-armed.")
    parser.add_argument("--detection-center-deadband", type=float, default=0.16, help="Center range where object approach drives straight.")
    parser.add_argument("--detection-prob-threshold", type=float, default=0.50)
    parser.add_argument("--detection-no-object-area-threshold", type=float, default=0.003)
    parser.add_argument("--detection-close-area-threshold", type=float, default=0.080)
    parser.add_argument("--detection-close-bbox-height-threshold", type=float, default=0.35)
    parser.add_argument("--detection-on-close-only", action="store_true", help="Deprecated; reports are now always saved only when close enough.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = PatrolConfig(
        model_path=args.model,
        image_size=args.image_size,
        speed=args.speed,
        approach_speed=args.approach_speed,
        turn_speed=args.turn_speed,
        turn_seconds=args.turn_seconds,
        inference_interval=args.inference_interval,
        confidence_threshold=args.confidence_threshold,
        smooth_window=args.smooth_window,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        device=args.device,
        dry_run=args.dry_run,
        detection_model_path=args.detection_model,
        detection_interval=args.detection_interval,
        detection_image_size=args.detection_image_size,
        detection_result_ttl=args.detection_result_ttl,
        require_fresh_detection_for_full_speed=not args.allow_full_speed_without_detection_result,
        detection_save_dir=args.detection_save_dir,
        detection_save_cooldown=args.detection_save_cooldown,
        detection_absence_required=args.detection_absence_required,
        detection_center_deadband=args.detection_center_deadband,
        detection_prob_threshold=args.detection_prob_threshold,
        detection_no_object_area_threshold=args.detection_no_object_area_threshold,
        detection_close_area_threshold=args.detection_close_area_threshold,
        detection_close_bbox_height_threshold=args.detection_close_bbox_height_threshold,
        detection_on_close_only=args.detection_on_close_only,
    )
    patrol = DrivingPatrol(config)
    signal.signal(signal.SIGTERM, patrol.request_stop)
    signal.signal(signal.SIGINT, patrol.request_stop)
    patrol.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
