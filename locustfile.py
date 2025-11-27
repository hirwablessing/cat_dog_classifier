from locust import HttpUser, task, between, events
import os
import cv2
import numpy as np

# Ensure we have a dummy image to test with
TEST_IMAGE_PATH = "test_image.jpg"

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    # Try to find a real image first
    possible_dirs = [
        os.path.join("data", "test", "cat"),
        os.path.join("data", "test", "dog"),
        os.path.join("data", "train", "cat"),
        os.path.join("data", "train", "dog")
    ]
    
    found = False
    for d in possible_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                import shutil
                shutil.copy(os.path.join(d, files[0]), TEST_IMAGE_PATH)
                print(f"Using real image for testing: {files[0]}")
                found = True
                break
    
    if not found and not os.path.exists(TEST_IMAGE_PATH):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(TEST_IMAGE_PATH, img)
        print(f"Created dummy test image at {TEST_IMAGE_PATH}")

class MLUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def predict_image(self):
        if os.path.exists(TEST_IMAGE_PATH):
            with open(TEST_IMAGE_PATH, "rb") as f:
                self.client.post("/predict", files={"file": f})
