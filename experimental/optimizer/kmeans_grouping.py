import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

communication_matrix = np.loadtxt("communication_matrix.txt")

inertia_values = []

# Define the range of k values you want to test
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(communication_matrix)
    inertia_values.append(kmeans.inertia_)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia_values, 'o-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

optimal_k = 4
kmeans_optimal = KMeans(n_clusters=optimal_k)
kmeans_optimal.fit(communication_matrix)
grouping_results = kmeans_optimal.labels_

print("Grouping Results:")
for device, group in enumerate(grouping_results):
    print(f"Device {device} is in group {group}")
