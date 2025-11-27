import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_image
import os

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
        else:
            print(f"Model not found at {self.model_path}")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise Exception("Model not loaded. Please train the model first.")
        
        processed_img = preprocess_image(image_path)
        prediction = self.model.predict(processed_img)
        
        score = prediction[0][0]
        
        if score > 0.5:
            label = "dog"
            confidence = score
        else:
            label = "cat"
            confidence = 1 - score
            
        return label, float(confidence)
