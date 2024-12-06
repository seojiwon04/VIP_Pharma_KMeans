import os
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip the header information
        f.read(16)
        # Read the rest of the file as a flat array of pixels
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = images.reshape(-1, 28*28)  # Reshape into 28x28 images
        return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# Load data from local files
X_train = load_mnist_images('./data/train-images-idx3-ubyte.gz')
true_labels = load_mnist_labels('./data/train-labels-idx1-ubyte.gz')

# Standardize the dataset
X_train = StandardScaler().fit_transform(X_train)
