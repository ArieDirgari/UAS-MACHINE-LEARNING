import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load model
model = load_model("batik_resnet50.h5")

CLASS_NAMES = ["kawung", "megamendung", "parang", "sidomukti", "truntum"]

st.title("Klasifikasi Motif Batik Indonesia")
st.write("Upload citra batik untuk diprediksi")

uploaded_file = st.file_uploader("Pilih citra batik", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Citra Input", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Hasil Prediksi")
    st.write(f"Motif: **{CLASS_NAMES[class_idx]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
