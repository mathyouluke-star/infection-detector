import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("🔍 Model Inference App")

# --- Model Selection ---
model_file = st.selectbox("Select model file", ["best.onnx", "best.torchscript"])
model = YOLO(model_file)

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display input image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save to temp file for YOLO inference
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)

        # 🔧 FIX: Force 416 for ONNX (since your model was exported at 416x416)
        if model_file.endswith(".onnx"):
            results = model(tmp.name, imgsz=416)
        else:
            results = model(tmp.name)

    # Save and display prediction
    results[0].save(filename="output.jpg")
    st.image("output.jpg", caption="Prediction", use_container_width=True)

    # Show detection details
    st.write("Detection Results:")
    st.json(results[0].tojson())
