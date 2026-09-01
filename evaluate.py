import pickle
import numpy as np

from data_loader import load_images
from nn import L_model_forward


TEST_DIR = "data/test"
TRAIN_DIR = "data/train"
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

    actual_cat = (Y_test == 0)
    actual_nocat = (Y_test == 1)

    predicted_cat = (predictions == 0)
    predicted_nocat = (predictions == 1)

    print()
    print("Confusion matrix:")
    print("-----------------")

    print("Actual cats:")
    print(f"  predicted cat:     {np.sum(actual_cat & predicted_cat)}")
    print(f"  predicted non-cat: {np.sum(actual_cat & predicted_nocat)}")

    print("Actual non-cats:")
    print(f"  predicted cat:     {np.sum(actual_nocat & predicted_cat)}")
    print(f"  predicted non-cat: {np.sum(actual_nocat & predicted_nocat)}")


if __name__ == "__main__":
    main()
