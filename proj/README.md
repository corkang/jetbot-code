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

## Timed Patrol Capture

JetBot should run only the Safety CNN. Object detection/classification should run later on the server or Raspberry Pi from saved patrol frames.

Run for 60 seconds and save about one frame per second:

```bash
cd ~/Notebook/English/expr/proj
python3 -u patrol_driving.py \
  --model ../models/safety_cnn.pt \
  --patrol-time 60 \
  --capture-interval 1.0 \
  --speed 0.25 \
  --turn-speed 0.25 \
  --confidence-threshold 0.60 \
  --device cpu
```

Frames are saved in a timestamped run folder:

```text
proj/img/patrol/YYYYMMDD_HHMMSS/
```

Each run folder contains:

- `0001.jpg`, `0002.jpg`, ...
- `metadata.json` with elapsed time, action, raw Safety CNN label, confidence, and probabilities for each saved frame

Copy a run folder from JetBot to a server:

```bash
rsync -avh --progress \
  jetbot@JETBOT_IP:~/Notebook/English/expr/proj/img/patrol/YYYYMMDD_HHMMSS/ \
  ~/jetbot_patrol/YYYYMMDD_HHMMSS/
```

This keeps JetBot lightweight: no detection model, no generated training data, and no batch object analysis on the robot.
