import json
import time
from pathlib import Path

import cv2


class LocalObjectReporter:
    """Report sink for detected objects.

    Today this saves a photo and metadata locally. Later this class can be
    replaced with a Raspberry Pi sender without changing patrol control logic.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report(self, frame, summary):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        area = int(summary["area_ratio"] * 1000.0)
        stem = "{}_area{:03d}".format(timestamp, area)
        image_path = self.output_dir / "{}.jpg".format(stem)
        metadata_path = self.output_dir / "{}.json".format(stem)

        cv2.imwrite(str(image_path), frame)
        metadata = {
            "timestamp": timestamp,
            "image_file": image_path.name,
            "state": summary.get("state"),
            "object_pixels": summary.get("object_pixels"),
            "area_ratio": summary.get("area_ratio"),
            "bbox": summary.get("bbox"),
            "bbox_height_ratio": summary.get("bbox_height_ratio"),
            "center_x_ratio": summary.get("center_x_ratio"),
            "max_probability": summary.get("max_probability"),
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return image_path, metadata_path
