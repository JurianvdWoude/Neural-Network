from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 64


def load_images(directory):
  """
  load images from ___/cat and ___/no-cat and convert them to the right size
  Returns:
    X: shape (12288, number of images)
    Y: shape (1, number of images)
  Labels:
    0: not a cat
    1: cat
  """
  directory = Path(directory)

  images = []
  labels = []

  for label_name, label in [("no-cat", 0), ("cat", 1)]:
    folder = directory / label_name

    for image_path in folder.iterdir():
      if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

      try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        image_array = np.asarray(image, dtype=np.float32)

        # Normalize pixels from 0 to 255 into 0 to 1
        image_array /= 255.0

        # Flatten array (64, 64, 3) -> (12288,)
        image_array = image_array.reshape(-1)

        images.append(image_array)
        labels.append(label)

      except Exception as e:
        print(f"Could not load {image_path}: {e}")

  X = np.array(images.T)
  Y = np.array(labels).reshape(1, -1)

  return X, Y
