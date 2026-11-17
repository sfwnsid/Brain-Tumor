import streamlit as st
import numpy as np

from keras.models import load_model
from skimage.transform import resize
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_brain_model():
    return load_model("brain_tumor_model.keras")

model = load_brain_model()

categories = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

with st.sidebar:
    st.header("🧠 About the App")
    st.write(
        "Upload a brain MRI scan and the model will classify it into one of four categories."
    )
    st.markdown("""
    **Supported classes**
    - **Glioma** – tumors arising from glial cells  
    - **Meningioma** – tumors of the meninges  
    - **No Tumor** – healthy / normal scan  
    - **Pituitary** – pituitary gland tumors  
    """)
    st.info("Model input size: **150 × 150** grayscale")
    st.caption("For educational / research purposes only.")

# Streamlit interface
st.title("Brain Tumor Classification")

st.write("Upload an MRI image for classification.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)


if uploaded_file is not None:


    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded MRI Image")
    img = np.array(image)
    img = resize(img, (150, 150, 1))
    img = np.expand_dims(img, axis=0)
    if st.button("Predict"):
        with st.spinner("Analyzing MRI scan…"):
            pred = model.predict(img, verbose=0)[0]
        
        res = np.argmax(pred[0])
        confidence = np.max(pred[0]) * 100
        st.subheader("Prediction Result")
        st.success(f"Class: {categories[res]}")
        st.write(f"Model confidence: {confidence:.2f}%")

        if confidence >= 80:
            st.success("High confidence prediction.")
        elif confidence >= 60:
            st.info("Moderate confidence – review recommended.")
        else:
            st.warning("Low confidence – consider consulting a medical specialist.")

else:
    st.info("Please upload an MRI image to begin.")
