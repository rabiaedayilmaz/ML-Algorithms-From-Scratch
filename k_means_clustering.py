import numpy as np
import matplotlib.pyplot as plt

#X = np.loadtxt("kmean_data.csv", delimiter=",")

# edit paths according to the folder :p
# gunah benden gitti yani sin go away away from me

image = plt.imread("test.jpg")
image = image / 255
X_img = np.reshape(image, (image.shape[0]*image.shape[1], 3))

#initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = 16

class KMeansClustering:
    def __init__(self, X, K, iterations=10):
        initial_centroids = self.init_kmean_centroids(X, K)
        centroids, idx = self.run_kmeans(X, initial_centroids, iterations)
        print("Shape of idx:", idx.shape)
        print("Closest centroid for the first five elements:", idx[:5])

        X_recovered = centroids[idx, :]
        X_recovered = np.reshape(X_recovered, image.shape)

        fig, ax = plt.subplots(1, 2, figsize=(4*2,4*2))

        ax[0].imshow(image)
        ax[0].set_title("Original")
        ax[0].set_axis_off()

        ax[1].imshow(X_recovered)
        ax[1].set_title(f"Compressed with {K} colors")
        ax[1].set_axis_off()

        plt.show()

    def init_kmean_centroids(self, X, K):
        randidx = np.random.permutation(X.shape[0])
        centroids = X[randidx[:K]]
        return centroids

    def find_closest_centroids(self, X, centroids):
        K = centroids.shape[0]
        idx = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distance = []
            for j in range(K):
                dist = np.sum(np.square(X[i] - centroids[j]))
                distance.append(dist)
            idx[i] = np.argmin(distance)

        return idx

    def compute_centroids(self, X, idx, K):
        m, n = X.shape
        centroids = np.zeros((K, n))
        for k in range(K):
            points = X[idx == k]
            centroids[k] = np.mean(points, axis=0)
        return centroids

    def run_kmeans(self, X, initial_centroids, max_iters=10):
        m, n = X.shape
        K = initial_centroids.shape[0]
        centroids = initial_centroids
        idx = np.zeros(m)

        for i in range(max_iters):
            print(f"K-Means Iteration: {i}/{max_iters}")
            idx = self.find_closest_centroids(X, centroids)
            centroids = self.compute_centroids(X, idx, K)

        return centroids, idx

#KMeansClustering(X, K)
KMeansClustering(X_img, K)