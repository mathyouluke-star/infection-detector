import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "🧫🦠 Infection Detector"
MODEL_CHOICES = ["best.onnx"]  # keep filenames as in repo root

# ---------------------- UI SETUP ----------------------
st.set_page_config(page_title="Infection Detector", page_icon="🧫", layout="wide")
st.title(APP_TITLE)
st.write(
    "Upload a microscope image (thin smear) to detect malaria-infected cells. "
    "The model runs fully in-browser on Streamlit Cloud."
)

# ---------------------- HELPERS -----------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load YOLO model once (ONNX or TorchScript)."""
    return YOLO(model_path)

def annotate_and_render(results, out_path: Path):
    # Save first result’s annotated image
    results[0].save(filename=str(out_path))
    # Build a small table from boxes (if available)
    try:
        boxes = results[0].boxes
        if boxes is not None:
            df = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            cls  = boxes.cls.cpu().numpy()  if hasattr(boxes.cls, "cpu")  else boxes.cls
            import pandas as pd
            tdf = pd.DataFrame(df, columns=["x1","y1","x2","y2"])
            tdf["conf"] = conf
            tdf["class"] = cls
            return tdf
    except Exception:
        pass
    return None

# ---------------------- SIDEBAR -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model file", MODEL_CHOICES)
    model_file = Path(model_name)
    if not model_file.exists():
        st.info(
            f"ℹ️ Put **{model_name}** in the repo root. "
            "You may keep only one of the two files and remove the other from MODEL_CHOICES."
        )
    else:
        st.success(f"✅ Model loaded: {model_name} ({model_file.stat().st_size/1_048_576:.2f} MB)")

# Lazy-load only when the file exists
model = load_model(model_file.as_posix()) if model_file.exists() else None

# ---------------------- MAIN AREA ---------------------
st.subheader("📤 Upload Image")
up = st.file_uploader("PNG, JPG, JPEG (≤ 10MB)", type=["png","jpg","jpeg"])

if up and model is not None:
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded", use_container_width=True)

    with st.spinner("Running inference…"):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            results = model(tmp.name)

    # Save annotated output and show side-by-side
    out_path = Path("prediction.jpg")
    det_table = annotate_and_render(results, out_path)

    with col2:
        st.image(str(out_path), caption="Prediction", use_container_width=True)
        if det_table is not None and len(det_table) > 0:
            st.caption("Detections")
            st.dataframe(det_table, use_container_width=True)
        else:
            st.info("No objects detected (or model returned no boxes).")

    # Optional raw JSON
    with st.expander("Raw result JSON"):
        st.json(results[0].tojson())
elif up and model is None:
    st.error("Model file not found in the repo. See README.md.")

st.markdown("---")
st.caption(
    "Tip: For larger models (>100MB), host weights on Hugging Face or Google Drive and download at runtime."
)

