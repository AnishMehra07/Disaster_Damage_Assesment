import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_yolo_boxes(label_file: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    with label_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    x1 = int(round((x_center - width / 2.0) * img_w))
    y1 = int(round((y_center - height / 2.0) * img_h))
    x2 = int(round((x_center + width / 2.0) * img_w))
    y2 = int(round((y_center + height / 2.0) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def majority_damage_class(mask_roi: np.ndarray) -> int:
    # xView2 convention (commonly): 0=background, 1=No Damage, 2=Minor, 3=Major, 4=Destroyed
    # Return 1..4 if present, else 1 (No Damage) as safe default when no damage pixels fall inside the box
    if mask_roi.size == 0:
        return 1
    values, counts = np.unique(mask_roi, return_counts=True)
    # Focus on {1,2,3,4}
    best_cls = 1
    best_cnt = -1
    for v, c in zip(values.tolist(), counts.tolist()):
        if v in (1, 2, 3, 4) and c > best_cnt:
            best_cls = v
            best_cnt = c
    if best_cnt <= 0:
        return 1
    return best_cls


def process_split(split: str) -> None:
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "dataset" / "images" / split
    labels_in_dir = base_dir / "dataset" / "labels" / split
    targets_dir = base_dir / "dataset" / "targets"
    labels_out_dir = base_dir / "dataset" / "labels_damage" / split
    os.makedirs(labels_out_dir, exist_ok=True)

    label_files = sorted([p for p in labels_in_dir.glob("*_post_disaster.txt")])
    print(f"Split '{split}': found {len(label_files)} post-disaster label files")

    converted = 0
    for label_path in label_files:
        stem = label_path.stem  # e.g., hurricane-florence_00000067_post_disaster
        image_path = images_dir / f"{stem}.png"
        target_path = targets_dir / f"{stem}_target.png"

        if not image_path.exists() or not target_path.exists():
            # Skip if required files are missing
            continue

        # Load image to get W,H
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Load target mask (grayscale)
        mask = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        boxes = load_yolo_boxes(label_path)
        out_lines: List[str] = []
        for _, xc, yc, w, h in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                # degenerate, skip
                continue
            roi = mask[y1:y2, x1:x2]
            dmg_cls = majority_damage_class(roi)  # 1..4
            # Map to 0..3 for YOLO classes per data.yaml order
            yolo_cls = dmg_cls - 1
            out_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Write out file (even if empty, to be explicit)
        out_path = labels_out_dir / f"{stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        converted += 1

    print(f"Split '{split}': wrote {converted} files to {labels_out_dir}")


def main() -> int:
    # Build new labels in dataset/labels_damage/{train,val}
    for split in ("train", "val"):
        process_split(split)
    print("Done. New labels are under 'dataset/labels_damage'.")
    print("Re-run your counter against 'dataset/labels_damage' to verify distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


