import pickle

from data_loader import load_images
from nn import L_layer_model


TRAIN_DIR = "data/train"
MODEL_PATH = "model/model.pkl"


def main():

    print("Loading training data...")

    X_train, Y_train = load_images(TRAIN_DIR)

    print(f"Training images: {X_train.shape[1]}")
    print(f"Input shape: {X_train.shape}")
    print(f"Labels shape: {Y_train.shape}")

    layers_dims = [
        64 * 64 * 3,
        20,
        7,
        5,
        1
    ]

    print("\nTraining neural network...")

    parameters, costs = L_layer_model(
        X_train,
        Y_train,
        layers_dims,
        learning_rate=0.0075,
        num_iterations=2000,
        print_cost=True
    )

    print("\nSaving model...")

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(parameters, file)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
