import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="Ridge Gourd Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Hybrid AI Plant Disease Detection System")
st.markdown(
    """
- Upload a Ridge Gourd leaf image (`JPG`, `JPEG`, or `PNG`).
- The system utilizes a **Hybrid DenseNet121 + XGBoost (Distilled)** architecture for high-accuracy diagnosis.
"""
)

# =========================================================
# Load Trained Artifacts
# =========================================================
MODEL_PATH = "hybridcnn_distilled_webapp_v2.h5"
IMG_SIZE = (224, 224)

# Exact classes extracted from your notebook
CLASS_NAMES = ['Healthy', 'Leaf_Minor_Infestation', 'Mosaic_Virus']

@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, True
    except Exception as e:
        return str(e), False

model, model_loaded = load_trained_model()

# =========================================================
# Sidebar — App Info
# =========================================================
st.sidebar.header("⚙️ Model Settings")
st.sidebar.info(
    "**Current Model:**\n\n"
    "Hybrid DenseNet121 Extractor + Distilled XGBoost Classifier (.h5)\n\n"
    "**Input Resolution:**\n\n"
    "224 x 224 pixels\n\n"
    "**Target Plant:**\n\n"
    "Ridge Gourd"
)

# =========================================================
# Tabs
# =========================================================
tab_prediction, tab_metrics = st.tabs(["🔍 Disease Prediction", "📊 Model Performance"])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab_prediction:
    st.markdown("### Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### Diagnosis Results")
            if not model_loaded:
                st.error(f"Failed to load model: {model}")
            else:
                with st.spinner("Analyzing leaf textures..."):
                    # Preprocess the image
                    image_rgb = image.convert('RGB')
                    img_resized = image_rgb.resize(IMG_SIZE)
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # DenseNet Preprocessing expects pixel values matching ImageNet distribution
                    img_processed = preprocess_input(img_array)
                    
                    # Predict
                    predictions = model.predict(img_processed)[0]
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_name = CLASS_NAMES[predicted_class_index]
                    
                    # Convert to percentage
                    confidence = float(predictions[predicted_class_index] * 100)
                    
                    # Display Metrics beautifully
                    st.metric(
                        label="Predicted Diagnosis",
                        value=predicted_class_name.replace("_", " ")
                    )
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.2f}%"
                    )
                    
                    st.markdown("### 📊 Class Probabilities")
                    # Format probability dictionary for Streamlit bar chart
                    prob_dict = {CLASS_NAMES[i].replace("_", " "): float(predictions[i] * 100) for i in range(len(CLASS_NAMES))}
                    st.bar_chart(prob_dict)

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab_metrics:
    st.markdown("### 🧠 Architecture Overview")
    st.write(
        "This application runs on a state-of-the-art hybrid architecture designed for agricultural pathology:"
    )
    st.markdown(
        """
        - **Feature Extractor:** DenseNet121 (Pre-trained on ImageNet). Captures complex, multi-scale venation and lesion patterns.
        - **Classifier:** XGBoost logic distilled into a sequential Keras Dense Layer. 
        - **Preprocessing:** Standard DenseNet spatial resolution scaling (224x224).
        """
    )
    
    st.markdown("### 📈 Evaluation Metrics (Ridge Gourd Dataset)")
    st.info("The deployed model achieved an accuracy of **over 91%** on the holdout validation set of 1,800 images during training.")
    
    col1, col2 = st.columns(2)
    # Extracted from the provided notebook log outputs
    col1.metric("XGBoost Pre-Distillation Accuracy", "91.17%") 
    col2.metric("Keras End-to-End Distilled Accuracy", "91.89%") 

    st.markdown("""
    **Distillation Note:** The original XGBoost classifier achieved 91.17% accuracy using DenseNet's flattened output. To allow this model to be deployed as a single, web-compatible `.h5` file, the XGBoost logic was "distilled" into native Keras Dense layers, successfully retaining its high predictive power without the need to deploy independent XGBoost and Keras endpoints.
    """)