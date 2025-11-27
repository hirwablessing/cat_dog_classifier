import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def create_data_generators(train_dir, test_dir):
    """
    Creates Keras ImageDataGenerators for training and testing.
    """
    # Check if directories exist and are not empty
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        raise ValueError(f"Train directory {train_dir} is missing or empty.")

    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )

    return train_data_gen, val_data_gen

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction.
    """
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0
    return img_array
