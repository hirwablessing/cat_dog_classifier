from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import Predictor
from src.train import train_model

app = FastAPI(title="Cat vs Dog Classifier API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

predictor = Predictor(MODEL_PATH)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        label, confidence = predictor.predict(temp_file)
        return {"filename": file.filename, "label": label, "confidence": confidence}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def run_retraining():
    try:
        print("Starting retraining task...")
        train_model(DATA_DIR, MODEL_PATH, epochs=15)
        predictor.load_model() # Reload model after training
        print("Retraining complete.")
    except Exception as e:
        import traceback
        print(f"Error during retraining: {e}")
        traceback.print_exc()

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_retraining)
    return {"message": "Retraining started in background"}

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...), label: str = "train"):
    # save the file to data/train/cat or dog based on filename or random    
    # Heuristic: if filename contains 'cat', put in cat folder, else dog
    category = "cat" if "cat" in file.filename.lower() else "dog"
    target_dir = os.path.join(DATA_DIR, 'train', category)
    os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": f"File saved to {category} training data"}
