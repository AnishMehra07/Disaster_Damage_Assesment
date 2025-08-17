import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import Overlay_Prediction

# App title
st.title("🛰️ AI Disaster Damage Assessment")
st.write("Upload pre and post-disaster images to detect and classify damage severity.")

# File uploads
before_img_file = st.file_uploader("Upload BEFORE disaster image", type=["jpg", "png", "jpeg"])
after_img_file = st.file_uploader("Upload AFTER disaster image", type=["jpg", "png", "jpeg"])

if before_img_file and after_img_file:
    # Sidebar: model and inference settings
    weights_path = st.sidebar.text_input("Weights path", value="runs/detect/train3/weights/best.pt")
    device_choice = st.sidebar.selectbox("Device", options=["auto", "cpu", "mps"], index=0)
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.01, 0.0088)
    iou_threshold = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)
    imgsz = st.sidebar.selectbox("Image size", [320, 512, 640, 800], index=2)
    use_tta = st.sidebar.checkbox("Test-time augmentation (TTA)", value=False)
    agnostic_nms = st.sidebar.checkbox("Class-agnostic NMS", value=False)
    max_det = st.sidebar.slider("Max detections", 10, 2000, 1000, 10)

    # Load YOLO model
    model = YOLO(weights_path)

    # Decode uploaded images
    before_img = cv2.imdecode(np.frombuffer(before_img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    after_img = cv2.imdecode(np.frombuffer(after_img_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run inference using predict API for consistent Results type
    results = model.predict(
        source=after_img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=None if device_choice == "auto" else device_choice,
        augment=use_tta,
        agnostic_nms=agnostic_nms,
        classes=None,
        max_det=max_det,
        verbose=False,
    )

    # Colors per class
    num_classes = len(results[0].names) if isinstance(results, list) and len(results) else 0
    default_colors = [(0,255,0), (0,255,255), (0,165,255), (0,0,255)]
    colors = default_colors * (max(1, num_classes) // len(default_colors) + 1)

    # Annotate predictions
    annotated_img = Overlay_Prediction(after_img.copy(), results, colors)

    # Display images
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB), caption="Before Disaster", use_container_width=True)
    col2.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="After Disaster (Detection)", use_container_width=True)

    # Damage summary
    damage_counts = {}
    if isinstance(results, list) and len(results) and getattr(results[0], "boxes", None) is not None:
        for cls_id in results[0].boxes.cls:
            cls_name = results[0].names[int(cls_id)]
            damage_counts[cls_name] = damage_counts.get(cls_name, 0) + 1

    st.subheader("📊 Damage Summary")
    if damage_counts:
        df = pd.DataFrame(list(damage_counts.items()), columns=["Damage Type", "Count"])
        st.dataframe(df)
        st.bar_chart(df.set_index("Damage Type"))
    else:
        st.info("No damage detections at current thresholds.")

    # Debug panel
    with st.expander("Debug: raw detections"):
        try:
            num_det = int(len(results[0].boxes)) if isinstance(results, list) and len(results) else 0
        except Exception:
            num_det = 0
        st.write({
            "weights": weights_path,
            "device": device_choice,
            "imgsz": imgsz,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "tta": use_tta,
            "agnostic_nms": agnostic_nms,
            "max_det": max_det,
            "num_detections": num_det,
        })
        if isinstance(results, list) and len(results) and getattr(results[0], "boxes", None) is not None and num_det > 0:
            boxes = results[0].boxes
            sample = min(10, num_det)
            det_preview = []
            for i in range(sample):
                try:
                    cls_idx = int(boxes.cls[i])
                    name = results[0].names.get(cls_idx, str(cls_idx))
                    conf = float(boxes.conf[i]) if boxes.conf is not None else None
                    xyxy = [int(v) for v in boxes.xyxy[i].tolist()]
                    det_preview.append({"cls": cls_idx, "name": name, "conf": conf, "xyxy": xyxy})
                except Exception:
                    continue
            st.write({"preview": det_preview})