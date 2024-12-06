import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as scikit_KMeans
import seaborn as sns
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.preprocessing import StandardScaler

'''
# Load MNIST data
data = scipy.io.loadmat('mnist_all.mat')

# Combine all training sets into a single data matrix
all_training_data = []
for i in range(10):
    key = f"train{i}"
    all_training_data.append(data[key])

# Stack all the training data into one array
all_training_data = np.vstack(all_training_data)

# Compute mean of each digit in the training set
T = []
for i in range(10):
    key = f"train{i}"
    T.append(np.mean(data[key], axis=0))
T = np.array(T)

# Display average images of each digit
plt.figure(figsize=(10, 5))
for i in range(10):
    digit = T[i]
    digit_image = digit.reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.rot90(np.flipud(digit_image), -1), cmap='gray')
    plt.axis('off')
plt.show()

# Find the digit in T with the smallest norm to the first test image of digit "4"
d = data['test4'][0].astype(float)
smallest_norm = float('inf')
index = -1

for i in range(10):
    current = np.linalg.norm(T[i] - d)
    if current < smallest_norm:
        smallest_norm = current
        index = i

print(f'smallest index: {index}')!
'''

# Define Euclidean distance function
def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))

# K-Means class definition
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def has_duplicate_centroids(self, centroids):
        unique_centroids = np.unique(centroids, axis=0)
        return len(unique_centroids) < len(centroids)

    def fit(self, X_train):
        print("Initializing centroids using k-means++ method...")
        # Initialize the centroids using k-means++ method
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters - 1):
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)  # Normalize the distances
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]
            self.centroids.append(X_train[new_centroid_idx])
        print("Centroids initialized.")

        iteration = 0
        prev_centroids = None
        print("Starting K-Means iterations...")
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Assign points to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids.copy()
            # Recalculate centroids as the mean of points in the cluster
            self.centroids = [np.mean(cluster, axis=0) if len(cluster) > 0 else prev_centroids[i] for i, cluster in enumerate(sorted_points)]

            # After updating centroids in your code
            if self.has_duplicate_centroids(self.centroids):
                print("Warning: Duplicate centroids detected.")
            
            iteration += 1
            print(f"Iteration {iteration} completed.")

        print("K-Means clustering finished.")

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

# Display cluster centers as images
def display(centers, title):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        digit_image = centers[i].reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(digit_image, cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# Load MNIST dataset using PyTorch
print("Loading MNIST dataset...")
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
X_train = mnist.data.numpy().reshape(-1, 28*28)  # Reshape images to 1D array of size 784
true_labels = mnist.targets.numpy()
print("MNIST dataset loaded successfully.")

# Run scikit k-means clustering on the combined data
print("Running scikit KMeans...")
kmeans_1 = scikit_KMeans(n_clusters=10, random_state=42)
kmeans_1.fit(X_train)
print("KMeans model fitted successfully.")

# Get the cluster centers
cluster_centers = kmeans_1.cluster_centers_
print("Model evaluation completed.")

# Fit KMeans model to the MNIST data
print("Running written KMeans model...")
kmeans_2 = KMeans(n_clusters=10, max_iter=200)
kmeans_2.fit(X_train)
print("KMeans model fitted successfully.")

# Evaluate the model
class_centers, classification = kmeans_2.evaluate(X_train)
print("Model evaluation completed.")

# Display cluster centers as images
print("Displaying scikit-KMeans...")
display(cluster_centers, 'scikit-KMeans')
print("Displaying written KMeans")
display(class_centers, 'written-KMeans')
