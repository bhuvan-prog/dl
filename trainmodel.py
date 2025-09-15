# =============================
# 1. Import Libraries
# =============================
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
import pickle


def load_dataset():
    with_mask_dir = 'data/with_mask'
    without_mask_dir = 'data/without_mask'
    
    withmask = os.listdir(with_mask_dir)
    withoutmask = os.listdir(without_mask_dir)

    # Labels
    withmask_labels = [1] * len(withmask)
    withoutmask_labels = [0] * len(withoutmask)
    labels = withmask_labels + withoutmask_labels

    # Preprocess Images
    data = []
    
    for img in withmask:
        image = Image.open(os.path.join(with_mask_dir, img))
        image = image.resize((120, 120))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    for img in withoutmask:
        image = Image.open(os.path.join(without_mask_dir, img))
        image = image.resize((120, 120))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    # Convert to NumPy Arrays
    x = np.array(data)
    y = np.array(labels)
    
    return x, y

# =============================
# 3. Build and Train Model
# =============================
def build_and_train_model():
    # Load dataset
    x, y = load_dataset()
    
    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    xscale = x_train / 255.0
    xtscale = x_test / 255.0

    # Build Model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(120,120,3)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(60, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='sigmoid')  # 1 unit for binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train Model
    print("Training model...")
    history = model.fit(xscale, y_train, epochs=5, validation_data=(xtscale, y_test))

    # Evaluate
    print("Test Accuracy:")
    test_loss, test_acc = model.evaluate(xtscale, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save Model
    model.save("mask_detector.h5")
    print("Model saved as mask_detector.h5")

    # Save as pickle file
    with open('mask_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as mask_detector.pkl")
    
    return model, test_acc

if __name__ == "__main__":
    build_and_train_model()