import io
import math
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ----------------- CONFIG -----------------
APP_TITLE = "🧫🦠 Infection Detector"
MODEL_PATH = "best.onnx"        # file at repo root
INPUT_SIZE = 416                # training/export size (adjust if different)
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASS_NAMES = ["infected", "normal"]  # update to your labels order
# ------------------------------------------


st.set_page_config(page_title="Infection Detector", page_icon="🧫", layout="wide")
st.title(APP_TITLE)
st.caption("Upload a microscope image to detect malaria-infected cells (ONNX, CPU).")

@st.cache_resource(show_spinner=False)
def load_sess(model_path: str):
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    return sess, in_name, out_names

def letterbox(im: Image.Image, new_size=416, color=(114,114,114)):
    """Resize & pad to square, return image and scaling meta."""
    w, h = im.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im_resized = im.resize((nw, nh))
    canvas = Image.new("RGB", (new_size, new_size), color)
    pad_w, pad_h = (new_size - nw) // 2, (new_size - nh) // 2
    canvas.paste(im_resized, (pad_w, pad_h))
    return canvas, scale, pad_w, pad_h, (w, h)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # x: (..., 4) [cx, cy, w, h] -> [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.45):
    """Simple NMS returning kept indices."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return np.array(keep, dtype=int)

def iou(b1, b2):
    # b1: (4,) , b2: (N,4)
    x1 = np.maximum(b1[0], b2[:,0])
    y1 = np.maximum(b1[1], b2[:,1])
    x2 = np.minimum(b1[2], b2[:,2])
    y2 = np.minimum(b1[3], b2[:,3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def postprocess(pred, orig_size, scale, pad_w, pad_h, conf_thres=0.25, iou_thres=0.45):
    """
    pred: (1, N, 5+nc) or (1, 5+nc, N)
    returns list of (x1,y1,x2,y2, conf, cls_id)
    """
    if pred.ndim != 3:
        raise ValueError(f"Unexpected output shape: {pred.shape}")
    if pred.shape[1] < pred.shape[2]:  # (1, N, 5+nc)
        pred = pred[0]
    else:  # (1, 5+nc, N) -> (N, 5+nc)
        pred = pred[0].transpose(1, 0)

    # YOLOv8 export typically gives: [x, y, w, h, conf, cls_scores...]
    if pred.shape[1] < 6:
        raise ValueError("Output does not look like YOLOv8 (need at least 6 columns).")

    boxes = xywh2xyxy(pred[:, :4])
    conf_obj = sigmoid(pred[:, 4:5])
    cls_scores = sigmoid(pred[:, 5:])
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores.max(axis=1, keepdims=True)
    conf = conf_obj * cls_conf

    # filter by confidence
    m = (conf[:, 0] >= conf_thres)
    boxes, conf, cls_ids = boxes[m], conf[m, 0], cls_ids[m]

    # undo letterbox
    boxes[:, [0,2]] -= pad_w
    boxes[:, [1,3]] -= pad_h
    boxes /= scale

    # clip to image size
    w, h = orig_size
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, w)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, h)

    if len(boxes) == 0:
        return []

    # NMS per class
    results = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms(boxes[idx], conf[idx], iou_thres=iou_thres)
        for i in idx[keep]:
            results.append((*boxes[i].tolist(), float(conf[i]), int(c)))
    return results

def draw_boxes(im: Image.Image, dets):
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for x1, y1, x2, y2, cf, cid in dets:
        draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=3)
        label = f"{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid}: {cf:.2f}"
        tw, th = draw.textlength(label, font=font), 12
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(34, 197, 94))
        draw.text((x1 + 3, y1 - th - 2), label, fill=(0, 0, 0), font=font)
    return im


# Sidebar controls
with st.sidebar:
    st.subheader("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90, CONF_THRES, 0.05)
    iou = st.slider("IOU threshold (NMS)", 0.10, 0.90, IOU_THRES, 0.05)
    st.caption("Edit class names in `app.py` if needed.")

# Load model session
if not Path(MODEL_PATH).exists():
    st.error(f"Model file `{MODEL_PATH}` not found in the repo root.")
else:
    sess, in_name, out_names = load_sess(MODEL_PATH)
    st.success(f"Model loaded ✓  ({MODEL_PATH})")

st.subheader("📤 Upload Image")
up = st.file_uploader("PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

if up and Path(MODEL_PATH).exists():
    img = Image.open(up).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)


    # Preprocess
    lb, scale, pad_w, pad_h, orig_size = letterbox(img, INPUT_SIZE)
    arr = np.asarray(lb, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW

    with st.spinner("Running inference…"):
        outputs = sess.run(None, {in_name: arr})
        pred = outputs[0]
        dets = postprocess(pred, orig_size, scale, pad_w, pad_h, conf, iou)

    # Draw and show
    out = img.copy()
    out = draw_boxes(out, dets)
    col1, col2 = st.columns(2)
    with col1:
        st.image(out, caption="Prediction", use_column_width=True)
    with col2:
        import pandas as pd
        if dets:
            df = pd.DataFrame(dets, columns=["x1","y1","x2","y2","conf","class_id"])
            df["class_name"] = df["class_id"].apply(lambda i: CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detections.")


