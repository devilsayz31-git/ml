# Chit 9: Implementation and Evaluation of the Elbow Method for Determining
# the Optimal Number of Clusters using Student Performance Dataset

# 1. Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Load Dataset
df = pd.read_csv("StudentsPerformance.csv")
X = df[['math score', 'reading score']].values
print("First 5 rows of the dataset:")
print(df[['math score', 'reading score']].head())
print("\nDataset Shape:", X.shape)

# 3. Visualize the Original Dataset
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], s=20)
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Original Dataset")
plt.grid(True)
plt.show()

# 4. Apply K-Means for Different Values of K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 5. Display WCSS Values
wcss_table = pd.DataFrame({
    'Number of Clusters (K)': range(1, 11),
    'WCSS': wcss
})
print("\nWCSS Values:")
print(wcss_table)

# 6. Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 7. Apply K-Means with Optimal K
# From the elbow curve, the elbow is approximately at K = 3.
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

# 8. Display Clustered Dataset
print("\nClustered Dataset:")
print(df[['math score', 'reading score', 'Cluster']].head(10))

# 9. Display Cluster Centers
centers = kmeans.cluster_centers_
centers_df = pd.DataFrame(centers, columns=['Math Score', 'Reading Score'])
print("\nCluster Centers:")
print(centers_df)

# 10. Visualize Final Clusters
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=clusters, s=20)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, label='Centroids')
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("K-Means Clustering with Optimal K = 3")
plt.legend()
plt.grid(True)
plt.show()

# 11. Final Result
print("================================================")
print("RESULT")
print("================================================")
print("Optimal number of clusters =", optimal_k)
print("K-Means clustering successfully implemented.")
print("The Elbow Method indicates that K = 3 is the optimal")
print("number of clusters for this dataset.")
