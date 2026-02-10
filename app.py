import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("waste_model.h5")

st.title("Organic vs Recyclable Classifier")

option = st.selectbox("Choose Input Mode", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Waste Image")

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (224,224))

        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Organic" if prediction < 0.5 else "Recyclable"

        st.image(img, channels="BGR")
        st.write("Prediction:", label)

else:
    cam = st.camera_input("Take a photo")

    if cam:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (224,224))

        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Organic" if prediction < 0.5 else "Recyclable"

        st.image(img, channels="BGR")
        st.write("Prediction:", label)
