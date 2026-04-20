import streamlit as st
import numpy as np
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load model
model = load_model("batik_resnet50.h5")

CLASS_NAMES = ["kawung", "megamendung", "parang", "sidomukti", "truntum"]

# Mapping motif -> provinsi (sesuai GeoJSON: state)
BATIK_ORIGIN = {
    "megamendung": "Jawa Barat",
    "kawung": "Yogyakarta",
    "parang": "Yogyakarta",
    "truntum": "Jawa Tengah",
    "sidomukti": "Jawa Tengah"
}

st.title("Klasifikasi Motif Batik Indonesia")
st.write("Upload citra batik untuk diprediksi")

uploaded_file = st.file_uploader("Pilih citra batik", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ================= IMAGE =================
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Citra Input", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ================= PREDICTION =================
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_class = CLASS_NAMES[class_idx]
    origin_prov = BATIK_ORIGIN[predicted_class]

    st.subheader("Hasil Prediksi")
    st.write(f"Motif: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    # ================= ASAL DAERAH =================
    st.subheader("Asal Daerah")
    st.write(f"Motif ini berasal dari: **{origin_prov}**")

    # ================= LOAD GEOJSON =================
    with open("data/indonesia.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # ================= MAP =================
    st.subheader("Peta Asal Motif Batik")

    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    # Highlight provinsi
    def style_function(feature):
        prov = feature["properties"]["state"]

        if prov.lower() == origin_prov.lower():
            return {
                "fillColor": "green",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.7
            }
        else:
            return {
                "fillColor": "gray",
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.2
            }

    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["state"],
            aliases=["Provinsi:"]
        )
    ).add_to(m)

    # ================= MARKER (OPSIONAL) =================
    if predicted_class == "megamendung":
        folium.Marker(
            location=[-6.737, 108.552],
            popup="Cirebon - Megamendung",
            icon=folium.Icon(color="green")
        ).add_to(m)

    # ================= LEGEND =================
    legend_html = """
    <div style="position: fixed; 
    bottom: 50px; left: 50px; width: 180px; height: 90px; 
    background-color: white; z-index:9999; font-size:14px;
    border:2px solid grey; padding: 10px;">
    <b>Legenda</b><br>
    <span style="color:green;">■</span> Asal Motif<br>
    <span style="color:gray;">■</span> Provinsi lain
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ================= SHOW MAP =================
    st_folium(m, width=700, height=500)