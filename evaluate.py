import pickle

import numpy as np

from data_loader import load_images
from nn import L_model_forward


TEST_DIR = "data/test"
MODEL_PATH = "model/model.pkl"


def main():

    print("Loading test data...")

    X_test, Y_test = load_images(TEST_DIR)

    with open(MODEL_PATH, "rb") as file:
        parameters = pickle.load(file)

    predictions, _ = L_model_forward(X_test, parameters)

    predictions = (predictions >= 0.5).astype(int)

    accuracy = np.mean(predictions == Y_test)

    print()
    print(f"Test images: {X_test.shape[1]}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
