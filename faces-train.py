import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

y_labels = []
x_train = []

for root, dir, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            # print(label_ids)
            # y_labels.append(label) # some number
            # x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
            pil_image = Image.open(path).convert("L")  # grayscale
            image_array = np.array(pil_image)
            print(image_array)