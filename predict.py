import pickle
import sys

import numpy as np
from PIL import Image

from nn import L_model_forward


IMAGE_SIZE = 64
MODEL_PATH = "model/model.pkl"


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.asarray(image, dtype=np.float32)
    image_array /= 255.0

    image_array = image_array.reshape(-1)

    # Network expects:
    # (input_size, number_of_examples)
    return image_array.reshape(-1, 1)


def predict(image_path):

    with open(MODEL_PATH, "rb") as file:
        parameters = pickle.load(file)

    X = prepare_image(image_path)

    prediction, _ = L_model_forward(X, parameters)

    probability = float(prediction[0, 0])

    if probability >= 0.5:
        label = "CAT"
        confidence = probability
    else:
        label = "DOG"
        confidence = 1 - probability

    return label, confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image>")
        sys.exit(1)

    image_path = sys.argv[1]

    label, confidence = predict(image_path)

    print()
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
