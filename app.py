import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("waste_model.h5")

st.set_page_config(page_title="Waste Classification", layout="centered")

st.title("♻️ Waste Classification System")
st.write("Classifies waste into **Organic** or **Recyclable**")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        label = "Organic Waste ♻️"
        confidence = (1 - prediction) * 100
    else:
        label = "Recyclable Waste 🔄"
        confidence = prediction * 100

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2f}%")
