from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DRIVING_LABELS = ["forward", "turn_left"]
DEFAULT_PREPROCESSING = "raw_rgb_normalized"
LAB_TABLE_PREPROCESSING = "lab_table_mask"


class Flatten(nn.Module):
    """Compatibility replacement for torch.nn.Flatten on old JetBot PyTorch."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class TinyDrivingCNN(nn.Module):
    """Small from-scratch CNN for JetBot safety/driving labels."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_conv_block(num_ch_in, num_ch_out, k, s, p):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=num_ch_in,
            out_channels=num_ch_out,
            kernel_size=k,
            stride=s,
            padding=p,
        ),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def make_fc_block(num_node_in, num_node_out, is_last_block):
    layers = [nn.Linear(in_features=num_node_in, out_features=num_node_out)]
    if not is_last_block:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


class ProfessorSafetyCNN(nn.Module):
    """CNN architecture used by Building CNN/CNN_safety_preprocess training."""

    def __init__(
        self,
        image_height=224,
        image_width=224,
        conv_hidden_channels=(8, 16, 32, 64, 128),
        fc_hidden_nodes=32,
        num_classes=2,
        img_ch=3,
    ):
        super().__init__()

        layers = []
        ch_in = img_ch
        feature_map_height = image_height
        feature_map_width = image_width

        for num_ch_out in conv_hidden_channels:
            layers.append(make_conv_block(ch_in, num_ch_out, 3, 1, 1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            feature_map_height = ((feature_map_height - 2) // 2) + 1
            feature_map_width = ((feature_map_width - 2) // 2) + 1
            ch_in = num_ch_out

        last_feature_pixels = feature_map_height * feature_map_width * conv_hidden_channels[-1]
        layers.append(Flatten())
        layers.append(make_fc_block(last_feature_pixels, fc_hidden_nodes, is_last_block=False))
        layers.append(make_fc_block(fc_hidden_nodes, num_classes, is_last_block=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def strip_module_prefix(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        clean[key] = value
    return clean


def _model_from_checkpoint_metadata(checkpoint, labels):
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    preprocessing = model_config.get("preprocessing", DEFAULT_PREPROCESSING)

    if "conv_hidden_channels" in model_config:
        model = ProfessorSafetyCNN(
            image_height=int(model_config.get("image_height", 224)),
            image_width=int(model_config.get("image_width", 224)),
            conv_hidden_channels=tuple(model_config.get("conv_hidden_channels", (8, 16, 32, 64, 128))),
            fc_hidden_nodes=int(model_config.get("fc_hidden_nodes", 32)),
            num_classes=len(labels),
        )
        return model, preprocessing

    return TinyDrivingCNN(num_classes=len(labels)), preprocessing


def load_driving_model(model_path, device, labels=DRIVING_LABELS):
    checkpoint = torch.load(model_path, map_location=device)
    labels = list(checkpoint.get("class_names", labels)) if isinstance(checkpoint, dict) else list(labels)
    model, preprocessing = _model_from_checkpoint_metadata(checkpoint, labels)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict checkpoint or dict containing state_dict/model_state_dict.")

    model.load_state_dict(strip_module_prefix(checkpoint))
    model.to(device)
    model.eval()
    metadata = {
        "labels": labels,
        "preprocessing": preprocessing,
    }
    return model, metadata


def _normalize_u8(channel):
    return channel.astype(np.float32) / 255.0


def _make_desk_mask(rgb):
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    lab_b = lab[:, :, 2]
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    _, lab_b_otsu = cv2.threshold(lab_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    wood_hue = ((hue >= 5) & (hue <= 40)).astype(np.uint8) * 255
    wood_sat = (saturation >= 18).astype(np.uint8) * 255
    bright_enough = (value >= 70).astype(np.uint8) * 255
    hsv_wood = cv2.bitwise_and(cv2.bitwise_and(wood_hue, wood_sat), bright_enough)

    desk_mask = cv2.bitwise_or(lab_b_otsu, hsv_wood)
    kernel = np.ones((5, 5), dtype=np.uint8)
    desk_mask = cv2.morphologyEx(desk_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    desk_mask = cv2.morphologyEx(desk_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return desk_mask


def _make_danger_channel(desk_mask):
    non_desk = cv2.bitwise_not(desk_mask)
    edges = cv2.Canny(desk_mask, 60, 160)
    danger = cv2.addWeighted(non_desk, 0.75, edges, 0.25, 0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(danger, kernel, iterations=1)


def _preprocess_lab_table_mask(frame, image_size, device):
    if frame.shape[0] != image_size or frame.shape[1] != image_size:
        frame = cv2.resize(frame, (image_size, image_size))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    desk_mask = _make_desk_mask(rgb)
    non_desk_mask = cv2.bitwise_not(desk_mask)
    danger_channel = _make_danger_channel(desk_mask)
    stacked = np.stack(
        [
            _normalize_u8(desk_mask),
            _normalize_u8(non_desk_mask),
            _normalize_u8(danger_channel),
        ],
        axis=0,
    )
    return torch.from_numpy(stacked).float().unsqueeze(0).to(device)


def preprocess_bgr_frame(frame, image_size, device, preprocessing=DEFAULT_PREPROCESSING):
    if preprocessing == LAB_TABLE_PREPROCESSING:
        return _preprocess_lab_table_mask(frame, image_size, device)
    if preprocessing != DEFAULT_PREPROCESSING:
        raise ValueError("Unsupported preprocessing mode: {}".format(preprocessing))

    if frame.shape[0] != image_size or frame.shape[1] != image_size:
        frame = cv2.resize(frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_label(
    model,
    frame,
    device,
    labels,
    image_size,
    preprocessing=DEFAULT_PREPROCESSING,
):
    x = preprocess_bgr_frame(
        frame,
        image_size=image_size,
        device=device,
        preprocessing=preprocessing,
    )
    logits = model(x)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy().flatten().tolist()
    best_index = int(np.argmax(probs))
    return labels[best_index], float(probs[best_index]), probs


def ensure_model_path(path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError("Model file not found: {}".format(model_path))
    return str(model_path)
