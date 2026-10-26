# Chit 7 (Set B): Implementation and Evaluation of K-Means Clustering using Mall Customer dataset

# 1. Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load Dataset
df = pd.read_csv("mall_customer.csv")
print("Original Dataset:")
print(df.head())

# 3. Check Dataset
print("\nShape of Dataset:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nFeatures Used for Clustering:")
print(X.head())

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data:")
print(X_scaled[:5])

# 6. Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

print("\nWCSS Values:")
for k, value in zip(range(1, 11), wcss):
    print("K =", k, "WCSS =", value)

# 7. Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()

# 8. Apply K-Means with K = 5 (elbow point)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 9. Display Clustered Dataset
print("\nClustered Dataset:")
print(df.head(10))

# 10. Display Cluster Centers
print("\nCluster Centers (Scaled):")
print(kmeans.cluster_centers_)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centers (Original Scale):")
print(centers)

# 11. Display Customers in Each Cluster
print("\nCustomers in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())

# 12. Calculate WCSS
wcss_value = kmeans.inertia_
print("\nWCSS =", wcss_value)

# 13. Calculate Silhouette Score
silhouette = silhouette_score(X_scaled, df['Cluster'])
print("\nSilhouette Score =", silhouette)

# 14. Silhouette Scores for Different K
silhouette_scores = []
for k in range(2, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

print("\nSilhouette Scores for Different K:")
for k, score in zip(range(2, 11), silhouette_scores):
    print("K =", k, "Silhouette Score =", score)

# 15. Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K")
plt.grid(True)
plt.show()

# 16. Visualize Clusters with Centroids
plt.figure(figsize=(8, 5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], s=50)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=250, label='Cluster Centers')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering with Cluster Centers")
plt.legend()
plt.grid(True)
plt.show()

# 17. Final Evaluation
print("\nFINAL K-MEANS EVALUATION")
print("Number of Clusters (K):", 5)
print("WCSS:", wcss_value)
print("Silhouette Score:", silhouette)
print("\nCluster Centers:")
print(centers)
