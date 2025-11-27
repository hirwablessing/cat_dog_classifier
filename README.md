# End-to-End Machine Learning Cycle - Cat vs Dog Classification

## Project Description
This project is an end-to-end Machine Learning pipeline for classifying images of cats and dogs. It demonstrates the complete ML lifecycle, including data processing, model training, API deployment, and a user-friendly interface.

**Dataset:** [Cat and Dog Dataset on Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

**Frontend:** [Hosted Frontend](https://cat-dog-ui.onrender.com)

**Backend:** [Hosted Backend](https://cat-dog-classifier-wp81.onrender.com/docs)

It includes:
- **Data Processing**: Automated ingestion and augmentation.
- **Model Training**: CNN model built with TensorFlow/Keras.
- **API**: FastAPI backend for serving predictions and handling retraining.
- **UI**: Streamlit dashboard for user interaction.
- **Deployment**: Dockerized application for easy setup.
- **Monitoring**: Load testing with Locust.

## Directory Structure
```
cat_dog_classifier/
│
├── notebook/           # Jupyter Notebook for model development
├── src/                # Source code for processing, training, and prediction
├── api/                # FastAPI application
├── ui/                 # Streamlit dashboard
├── data/               # Dataset directory (train/test)
├── models/             # Saved models
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Container orchestration
└── locustfile.py       # Load testing script
```

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed.
- Python 3.10+ (if running locally without Docker).

### 1. Run locally with Docker
This will start the API and UI services.
```bash
docker-compose up --build
```
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### 2. Run Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the API:
```bash
uvicorn api.main:app --reload
```

Start the UI (in a separate terminal):
```bash
streamlit run ui/app.py
```

## Usage

### Prediction
1. Go to the **UI** (http://localhost:8501).
2. Navigate to the **Prediction** page.
3. Upload an image of a cat or dog.
4. Click "Predict" to see the result.

### Retraining
1. Go to the **Retraining** page in the UI.
2. Upload new images.
3. Click "Start Retraining Pipeline".
4. The model will retrain in the background and update automatically.

## Video Demo
https://youtu.be/hpKFiA5Skuk