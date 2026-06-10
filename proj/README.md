# JetBot Patrol Runtime

This folder contains plain Python runtime code for the desk patrol project. It follows the official JetBot notebook patterns:

- `Camera.instance(width=224, height=224)` for low-memory camera input.
- `Robot.set_motors(...)` for direct motor control.
- Low-rate polling instead of running CNN inference on every camera frame.

## Current Runtime

`patrol_driving.py` connects the first safety/driving CNN only.

Labels:

```text
forward
turn_left
```

Policy:

- `forward`: move both wheels at `0.4`.
- `turn_left`: rotate in place briefly using left wheel `-0.4`, right wheel `0.4`.
- low model confidence: treat as `turn_left`.

## Run

From JetBot:

```bash
cd ~/Notebook/English/expr/proj
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --device cpu
```

Conservative first run:

```bash
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --speed 0.25 --turn-speed 0.25 --device cpu
```

Load model only:

```bash
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --dry-run --device cpu
```

## Model Format

`--model` can point to either:

- older `TinyDrivingCNN` checkpoints
- `CNN_safety_preprocess/checkpoint/safety_cnn.pt`

Accepted checkpoint formats:

- raw `state_dict`
- `{"state_dict": ...}`
- `{"model_state_dict": ...}`
- `{"model_state_dict": ..., "model_config": ..., "class_names": ...}`

The model architecture and preprocessing are defined in `driving_model.py`.
If the checkpoint has `model_config.preprocessing == "lab_table_mask"`, the runtime uses the same desk-mask preprocessing used during local training.

## Use The Preprocess Safety CNN

On the Mac, after training:

```bash
cd "/Users/corkang/Desktop/Course-code/7_SpecialTopics_AIConvergence/cv-project-jetbot"
/Users/corkang/Desktop/Course-code/7_SpecialTopics_AIConvergence/cv-class/.venv/bin/python "Building CNN/CNN_safety_preprocess/convert_checkpoint_for_jetbot.py"
```

Do not copy the training checkpoint directly to JetBot. Newer PyTorch saves `.pt` files in zip serialization format, which old JetBot/Python 3.6 PyTorch may fail to read with `tarfile.InvalidHeaderError`. The converter writes a legacy-format deployment checkpoint at:

```text
jetbot-code/models/safety_cnn.pt
```

Then push/pull `jetbot-code` using your normal workflow.

On JetBot:

```bash
cd ~/Notebook/English/expr
git pull
cd proj
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --dry-run --device cpu
```

If dry-run loads successfully:

```bash
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --speed 0.25 --turn-speed 0.25 --confidence-threshold 0.60 --device cpu
```

After a cautious first test, increase speed only if behavior is stable:

```bash
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --speed 0.4 --turn-speed 0.4 --confidence-threshold 0.60 --device cpu
```

## Next Extension Point

Object detection/reporting should be added as a separate module that reads the same camera frame after the driving loop is stable. Keep CNN1 driving safety authoritative over report behavior.

## Run With Object Detection

Place the converted detection checkpoint at:

```text
jetbot-code/models/detection_seg.pt
```

On JetBot, first load both models without opening camera or motors:

```bash
cd ~/Notebook/English/expr/proj
python3 -u patrol_driving.py --model ../models/safety_cnn.pt --detection-model ../models/detection_seg.pt --dry-run --device cpu
```

Then run cautiously:

```bash
python3 -u patrol_driving.py \
  --model ../models/safety_cnn.pt \
  --detection-model ../models/detection_seg.pt \
  --speed 0.25 \
  --turn-speed 0.25 \
  --confidence-threshold 0.60 \
  --detection-interval 1.0 \
  --detection-image-size 160 \
  --detection-result-ttl 8.0 \
  --detection-save-cooldown 8.0 \
  --device cpu
```

Detection runs only when the safety CNN action is `forward`, and only at the configured low-rate interval. The segmentation CNN runs in a background worker so slow detection inference does not block the safety driving loop. While there is no fresh detection result yet, the robot uses `forward_slow` instead of full-speed `forward`. The robot stops only when a close-enough object is being reported.

If detection is slow on JetBot, first try `--detection-image-size 160`. If it is still slow, try `--detection-image-size 128`. This is possible because the segmentation CNN is fully convolutional, but smaller inputs may reduce mask quality.

Object behavior:

- `No object`: normal safety CNN patrol.
- `Object detected (far)`: enter slow approach mode. The robot turns left/right briefly if the object is off-center, otherwise it moves forward slowly.
- `Object detected (close enough)`: stop, save report image and metadata, then suppress duplicate reports.
- After reporting once, reporting is re-armed only after the object disappears from the camera for several detection checks.

Detected reports are saved under:

```text
proj/img/detected/
```

Photos are saved only after the segmentation mask reaches the close-enough threshold. Each report includes both `.jpg` and `.json` files so the local reporter can later be replaced with a Raspberry Pi sender.

Do not copy `Building CNN/CNN_detection/data/generated/` to JetBot. Only copy the final checkpoint.
