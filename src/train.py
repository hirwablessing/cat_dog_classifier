import os
import tensorflow as tf
from src.model import create_model
from src.preprocessing import create_data_generators
from src.utils import get_data_paths

def train_model(base_data_path, model_save_path, epochs=15):
    
    train_dir, test_dir = get_data_paths(base_data_path)
    
    train_gen, val_gen = create_data_generators(train_dir, test_dir)
    
    model = create_model()
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return history

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
    
    train_model(DATA_DIR, MODEL_PATH)
