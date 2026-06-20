import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("outputs/elbow_method.png")
plt.close()

# KMeans Model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters

# Visualization
plt.figure(figsize=(8,6))

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=clusters
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    marker='X'
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.savefig("outputs/customer_clusters.png")
plt.close()

# Save Summary
with open("outputs/cluster_summary.txt", "w") as f:
    f.write("Customer Segmentation Summary\n")
    f.write("="*40 + "\n\n")

    for i in range(5):
        count = len(df[df['Cluster'] == i])
        f.write(f"Cluster {i}: {count} customers\n")

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

print("\nProject Completed Successfully")