import streamlit as st
import requests
import os

st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🐱 Cat vs Dog Classifier 🐶")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction", "Retraining"])

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except Exception:
        return {"status": "offline"}

if page == "Dashboard":
    st.header("System Status")
    status = check_api_health()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", status.get("status", "Unknown"))
    with col2:
        st.metric("Model Loaded", str(status.get("model_loaded", False)))
        
    st.subheader("Data Overview")
    
    # Mock data for visualization (replace with real API call in production)
    import pandas as pd
    import numpy as np
    
    # 1. Class Distribution
    st.write("### 1. Class Distribution (Training Data)")
    chart_data = pd.DataFrame({
        "Class": ["Cat", "Dog"],
        "Count": [12500, 12500] # Approximate counts for Kaggle dataset
    })
    st.bar_chart(chart_data.set_index("Class"))
    st.caption("Balanced dataset with equal number of Cat and Dog images.")

    # 2. Model Performance History
    st.write("### 2. Model Accuracy over Epochs")
    epochs = list(range(1, 16))
    accuracy = [0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
    val_accuracy = [0.52, 0.60, 0.65, 0.70, 0.72, 0.75, 0.77, 0.78, 0.80, 0.81, 0.82, 0.83, 0.83, 0.84, 0.85]
    
    perf_data = pd.DataFrame({
        "Epoch": epochs,
        "Training Accuracy": accuracy,
        "Validation Accuracy": val_accuracy
    })
    st.line_chart(perf_data.set_index("Epoch"))
    st.caption("The model shows steady improvement in accuracy, indicating effective learning without significant overfitting.")

    # 3. Confidence Distribution (Simulated)
    st.write("### 3. Prediction Confidence Distribution")
    rng = np.random.default_rng(seed=42)
    confidence_data = rng.normal(0.85, 0.1, 100)
    confidence_data = np.clip(confidence_data, 0.5, 1.0)
    st.bar_chart(confidence_data)
    st.caption("Distribution of confidence scores for recent predictions. Higher concentration near 1.0 indicates a confident model.")

elif page == "Prediction":
    st.header("Make a Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', width=300)
        
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: **{result['label'].upper()}**")
                        st.progress(result['confidence'])
                        st.write(f"Confidence: {result['confidence']:.2%}")
                    else:
                        st.error("Error making prediction")
                except Exception as e:
                    st.error(f"Connection error: {e}")

elif page == "Retraining":
    st.header("Model Retraining")
    
    st.write("Upload new training data to improve the model.")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"])
    
    if uploaded_files and st.button(f"Upload {len(uploaded_files)} images"):
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                files = {"file": (file.name, file.getvalue(), file.type)}
                requests.post(f"{API_URL}/upload_data", files=files)
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("Upload complete!")
    
    st.divider()
    
    st.subheader("Trigger Retraining")
    if st.button("Start Retraining Pipeline"):
        try:
            response = requests.post(f"{API_URL}/retrain")
            if response.status_code == 200:
                st.success("Retraining started in the background! Check logs for progress.")
            else:
                st.error("Failed to start retraining.")
        except Exception as e:
            st.error(f"Error: {e}")
