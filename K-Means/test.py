import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


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

# Display cluster centers as images
def display(centers, title):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        digit_image = centers[i].reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(digit_image, cmap='gray_r')
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


# Run PCA for dimensionality reduction
clus_dataset = StandardScaler().fit_transform(X_train)
pca = PCA(0.98) # covered variance is 98%
pca.fit(clus_dataset)

print("Number of components before PCA  = " + str(X_train.shape[1]))
print("Number of components after PCA 0.98 = " + str(pca.n_components_))

clus_dataset = pca.transform(clus_dataset)

'''
# Step 1: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_train)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # For visualization purposes
data_pca = pca.fit_transform(data_scaled)

# Step 3: Visualize explained variance ratio
print("Visualizing variance...")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Ratio')
plt.show()

# Step 4: Determine optimal k using WCSS and Silhouette
print("determining optimal k using WCSS and Silhouette scores...")
wcss = []
silhouette_scores = []

for k in range(2, 11):
    print("for k of " + str(k))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_pca, kmeans.labels_))

print("done")

# Elbow Method Plot
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Silhouette Score Plot
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

'''
# Run scikit k-means clustering on the combined data
print("Running scikit KMeans...")
kmeans_1 = KMeans(n_clusters=10, random_state=42)
kmeans_1.fit(clus_dataset)
print("KMeans model fitted successfully.")

# Get the cluster centers
cluster_centers = kmeans_1.cluster_centers_
print("Model evaluation completed.")

# Display cluster centers as images
print("Displaying scikit-KMeans...")
original_centers = pca.inverse_transform(cluster_centers)
display(original_centers, 'scikit-KMeans with PCA')







