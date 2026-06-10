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
python3 patrol_driving.py --model ../models/safety_cnn.pt
```

Conservative first run:

```bash
python3 patrol_driving.py --model ../models/safety_cnn.pt --speed 0.25 --turn-speed 0.25
```

Load model only:

```bash
python3 patrol_driving.py --model ../models/safety_cnn.pt --dry-run
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
mkdir -p jetbot-code/models
cp "Building CNN/CNN_safety_preprocess/checkpoint/safety_cnn.pt" "jetbot-code/models/safety_cnn.pt"
```

Then push/pull `jetbot-code` using your normal workflow.

On JetBot:

```bash
cd ~/Notebook/English/expr
git pull
cd proj
python3 patrol_driving.py --model ../models/safety_cnn.pt --dry-run
```

If dry-run loads successfully:

```bash
python3 patrol_driving.py --model ../models/safety_cnn.pt --speed 0.25 --turn-speed 0.25 --confidence-threshold 0.60
```

After a cautious first test, increase speed only if behavior is stable:

```bash
python3 patrol_driving.py --model ../models/safety_cnn.pt --speed 0.4 --turn-speed 0.4 --confidence-threshold 0.60
```

## Next Extension Point

Object detection/reporting should be added as a separate module that reads the same camera frame after the driving loop is stable. Keep CNN1 driving safety authoritative over report behavior.
