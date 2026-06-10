from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DETECTION_STATES = {
    "none": "No object",
    "far": "Object detected (far)",
    "close": "Object detected (close enough)",
}


def make_conv_block(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
    )


class SegmentationCNN(nn.Module):
    """Binary object segmentation CNN trained in Building CNN/CNN_detection."""

    def __init__(self, img_ch=3, num_classes=2):
        super().__init__()
        self.enc1 = make_conv_block(img_ch, 16)
        self.enc2 = make_conv_block(16, 32)
        self.enc3 = make_conv_block(32, 64)
        self.bottleneck = make_conv_block(64, 128)
        self.dec3 = make_conv_block(128 + 64, 64)
        self.dec2 = make_conv_block(64 + 32, 32)
        self.dec1 = make_conv_block(32 + 16, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, images):
        e1 = self.enc1(images)
        x = F.max_pool2d(e1, kernel_size=2, stride=2)
        e2 = self.enc2(x)
        x = F.max_pool2d(e2, kernel_size=2, stride=2)
        e3 = self.enc3(x)
        x = F.max_pool2d(e3, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.head(x)


def ensure_detection_model_path(path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError("Detection model file not found: {}".format(model_path))
    return str(model_path)


def load_detection_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    num_classes = int(model_config.get("num_classes", 2))
    model = SegmentationCNN(img_ch=3, num_classes=num_classes)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Expected a detection checkpoint dict.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    postprocess = checkpoint.get("postprocess_config", {}) if isinstance(checkpoint, dict) else {}
    metadata = {
        "image_height": int(model_config.get("image_height", 224)),
        "image_width": int(model_config.get("image_width", 224)),
        "class_names": list(checkpoint.get("class_names", ["background", "object"]))
        if isinstance(checkpoint, dict)
        else ["background", "object"],
        "postprocess_config": postprocess,
    }
    return model, metadata


def preprocess_detection_frame(frame, image_width, image_height, device):
    if frame.shape[1] != image_width or frame.shape[0] != image_height:
        frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).to(device)


def summarize_object_mask(mask, image_width, image_height, no_object_area_threshold, close_area_threshold, close_bbox_height_threshold):
    object_pixels = int((mask > 0).sum())
    area_ratio = float(object_pixels) / float(image_width * image_height)
    if object_pixels <= 0 or area_ratio < no_object_area_threshold:
        return {
            "state": DETECTION_STATES["none"],
            "object_pixels": object_pixels,
            "area_ratio": area_ratio,
            "bbox": None,
            "bbox_height_ratio": 0.0,
            "center_x_ratio": None,
        }

    ys, xs = np.where(mask > 0)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    bbox_height_ratio = float(bbox[3] - bbox[1] + 1) / float(image_height)
    center_x_ratio = float(xs.mean()) / float(image_width)
    if area_ratio >= close_area_threshold or bbox_height_ratio >= close_bbox_height_threshold:
        state = DETECTION_STATES["close"]
    else:
        state = DETECTION_STATES["far"]

    return {
        "state": state,
        "object_pixels": object_pixels,
        "area_ratio": area_ratio,
        "bbox": bbox,
        "bbox_height_ratio": bbox_height_ratio,
        "center_x_ratio": center_x_ratio,
    }


@torch.no_grad()
def detect_object(
    model,
    frame,
    device,
    image_width,
    image_height,
    prob_threshold=0.50,
    no_object_area_threshold=0.003,
    close_area_threshold=0.080,
    close_bbox_height_threshold=0.35,
):
    x = preprocess_detection_frame(frame, image_width, image_height, device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
    mask = (probs >= prob_threshold).astype(np.uint8)
    summary = summarize_object_mask(
        mask,
        image_width=image_width,
        image_height=image_height,
        no_object_area_threshold=no_object_area_threshold,
        close_area_threshold=close_area_threshold,
        close_bbox_height_threshold=close_bbox_height_threshold,
    )
    summary["max_probability"] = float(probs.max()) if probs.size else 0.0
    return summary, mask
