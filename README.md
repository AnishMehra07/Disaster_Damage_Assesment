## Disaster Damage Assessment (YOLOv8 + Streamlit)

This project detects and classifies building damage severity from post-disaster satellite imagery using Ultralytics YOLOv8 and provides an interactive Streamlit app for inference and visualization.

### Key features
- **Damage detection and severity classification** into 4 classes: `No Damage`, `Minor Damage`, `Major Damage`, `Destroyed`.
- **Interactive Streamlit UI**: upload before/after images, visualize detections, tune thresholds, and inspect raw outputs.
- **Training utilities**: dataset conversion, splitting, label validation, and label counting.

---

## Project structure

```
Disaster_Damage _Assesment_copy/
  app.py                      # Streamlit app for inference and visualization
  utils.py                    # Drawing overlays and helper logic
  requirements.txt            # Python dependencies
  dataset/
    data.yaml                 # YOLO dataset config (4 classes)
    images/{train,val}/       # Images in YOLO format
    labels/{train,val}/       # Labels in YOLO TXT format
    targets/                  # Optional reference masks/targets
  runs/detect/train*/weights/ # Trained YOLOv8 weights (best.pt, last.pt)
  convert.py                  # Convert JSON WKT labels to YOLO TXT
  split_dataset.py            # Split data into train/val
  count_labels.py             # Count label distribution per class
  check_for_empty.py          # Find empty/malformed labels
  toYOLO.py                   # Additional conversion helpers
  assign_damage_from_targets.py # Optional damage assignment from masks
```

---

## Setup

### 1) Create environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on macOS and want to try Apple Silicon acceleration, also install `torch` with MPS support (see the official PyTorch install guide). The app supports selecting `cpu` or `mps` at runtime.

### 2) Verify dataset layout
- Ensure images are under `dataset/images/{train,val}`.
- Ensure labels are under `dataset/labels/{train,val}` with one `.txt` per image (YOLO format).
- Confirm `dataset/data.yaml` contains:
```yaml
nc: 4
names: ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
```

---

## Training

You can train with the Ultralytics CLI (recommended) or a script. Example CLI command:
```bash
pip install ultralytics
yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 project=runs name=train3
```

This will produce weights at `runs/detect/train3/weights/best.pt`.

Helpful utilities:
- `count_labels.py` to see class balance:
```bash
python count_labels.py --labels-train dataset/labels/train --labels-val dataset/labels/val --data-yaml dataset/data.yaml
```
- `split_dataset.py` to split images/labels into train/val.
- `check_for_empty.py` to locate empty or malformed labels.
- `convert.py` / `toYOLO.py` for converting external annotations to YOLO format.

---

## Inference (Streamlit app)

### Launch
```bash
streamlit run app.py
```

### Usage
1. Upload a BEFORE and an AFTER image. The model runs on the AFTER image.
2. In the sidebar, set:
   - **Weights path**: e.g., `runs/detect/train3/weights/best.pt`.
   - **Confidence threshold**: start low (0.05–0.15) for weak detections.
   - **IoU threshold**: ~0.45 is a good default.
   - **Image size**: 640–800 for better recall (higher is slower).
   - **Device**: `cpu` or `mps` (macOS). If scores seem unstable, try `cpu`.
   - **TTA** and **Class-agnostic NMS**: can help surface borderline cases.
3. The app shows side‑by‑side BEFORE/AFTER and draws detections with labels and confidences.
4. Open the “Debug: raw detections” expander to inspect counts and a preview of raw outputs.

---

## Batch inference (quick test)

Run a one‑off prediction in Python to verify your weights outside the app:
```bash
python - << 'PY'
from ultralytics import YOLO
model = YOLO('runs/detect/train3/weights/best.pt')
res = model.predict('dataset/images/val/guatemala-volcano_00000006_post_disaster.png', conf=0.1, iou=0.45, imgsz=640, verbose=False)
r0 = res[0]
print('Detections:', len(r0.boxes) if getattr(r0,'boxes',None) is not None else 0)
PY
```

---

## Troubleshooting

- **No detections, but there is visible damage**
  - Lower the confidence to 0.05–0.10 and increase image size to 640–800.
  - Enable TTA and/or Class‑agnostic NMS.
  - Confirm the weights path points to the intended model.
  - Verify your model was trained on the same 4‑class schema as `dataset/data.yaml`.
  - Try `Device=cpu` on macOS; `mps` can sometimes yield different scores.
  - Use the debug expander to confirm raw detections exist (even if filtered by thresholds).

- **Streamlit deprecation about image width**
  - The app uses `use_container_width=True` to avoid deprecated params.

- **TypeError: Tensor round / dtype issues**
  - The overlay function converts tensors to Python numbers internally; ensure you’re on the updated `utils.py`.

- **Missing dependencies in editor (lint warnings)**
  - Ensure you’ve activated the virtual environment and installed requirements.

---

## Notes on labels and conversions

- Labels follow YOLO format: `<class_id> <x_center> <y_center> <width> <height>` normalized to [0,1].
- `convert.py` can convert JSON with WKT polygons to YOLO bboxes; adjust class mapping if needed.
- Use `check_for_empty.py` to find any labels with zero or invalid lines.

---

## Acknowledgements

- Built with **Ultralytics YOLOv8** (`ultralytics`).
- Streamlit for the UI.
- Inspired by disaster response datasets such as xBD.

---

## License

This repository is intended for research and educational use. Please verify and comply with the licenses of any datasets and third‑party models you use.


