import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model_path = 'Model/FindingWaldo.pt'
model = YOLO(model_path)

# Streamlit interface
st.title("YOLOv8 Object Detection")
st.write("Upload an image and the model will perform object detection.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting...")

    # Perform object detection
    results = model(image)

    # Extract the first result from the list
    result = results[0]

    # Convert the result image to a format suitable for display
    # result.plot() returns a matplotlib image with altered channels, so we need to ensure it's in the correct format.
    result_img = result.plot()  # This returns an image with bounding boxes

    # Ensure correct color format (from BGR to RGB)
    result_img_rgb = Image.fromarray(result_img[..., ::-1])  # Convert from BGR to RGB

    # Display the results
    st.image(result_img_rgb, caption="Detected Image", use_column_width=True)

    # Optionally display the raw results as a DataFrame (requires pandas)
    if result.boxes:
        st.write(result.boxes.data.cpu().numpy())  # Convert to numpy for better display





