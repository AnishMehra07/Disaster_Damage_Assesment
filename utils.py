

import cv2
import numpy as np

def Overlay_Prediction(img, results, colors):
    
    if isinstance(results, list):
        if len(results) == 0:
            return img
        results = results[0]

    
    if getattr(results, "boxes", None) is None or len(results.boxes) == 0:
        return img

    
    names = getattr(results, "names", {})
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        box_vals = box.tolist() if hasattr(box, "tolist") else list(box)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box_vals]
        cls_idx = int(cls.item() if hasattr(cls, "item") else cls)
        color = colors[cls_idx % len(colors)] if colors else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = str(names.get(cls_idx, str(cls_idx)))
        if conf is not None:
            try:
                conf_val = float(conf.item() if hasattr(conf, "item") else conf)
                label = f"{label} {conf_val:.2f}"
            except Exception:
                pass
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

