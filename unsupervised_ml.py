import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([[15,39],[16,81],[17,6],[18,77],[19,40],[20,76],[55,35],[57,82],[58,5],[60,78],[62,40],[65,77]])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print(f"Cluster labels: {labels}")
print(f"Centroids: {kmeans.cluster_centers_}")
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Unsupervised Learning: KMeans Clustering')
plt.legend()
plt.savefig('unsupervised_plot.png')
plt.show()