import seaborn as sns
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset using PyTorch
print("Loading MNIST dataset...")
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
X_train = mnist.data.numpy().reshape(-1, 28*28)  # Reshape images to 1D array of size 784
true_labels = mnist.targets.numpy()
print("MNIST dataset loaded successfully.")

# Standardize the dataset
print("Standardizing the dataset...")
X_train = StandardScaler().fit_transform(X_train)
print("Dataset standardized.")

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

# Fit KMeans model to the MNIST data
print("Fitting KMeans model...")
kmeans = KMeans(n_clusters=10, max_iter=100)
kmeans.fit(X_train)
print("KMeans model fitted successfully.")

# Evaluate the model
print("Evaluating the KMeans model...")
class_centers, classification = kmeans.evaluate(X_train)
print("Model evaluation completed.")

# Display cluster centers as images
plt.figure(figsize=(10, 5))
for i in range(10):
    digit_image = class_centers[i].reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.rot90(np.flipud(digit_image), -1), cmap='gray')
    plt.title(f'Cluster {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Dimensionality reduction for visualization using PCA
print("Reducing dimensionality using PCA...")
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
print("Dimensionality reduction completed.")

# Visualize a subset of the data and the centroids
print("Visualizing data and centroids...")
sns.scatterplot(x=X_train_2D[:3000, 0], y=X_train_2D[:3000, 1], hue=classification[:3000], palette="deep", legend=None)
centroids_2D = pca.transform(kmeans.centroids)
plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], color='red', marker='x', s=200)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
print("Visualization completed.")

