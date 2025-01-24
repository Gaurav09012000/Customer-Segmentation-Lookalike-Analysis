import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
customers_df = pd.read_csv(r"C:\Users\gaurgaut\Downloads\Customers.csv")
transactions_df = pd.read_csv(r"C:\Users\gaurgaut\Downloads\Transactions.csv")

# Step 2: Merge customer and transaction data
merged_df = pd.merge(customers_df, transactions_df, on="CustomerID", how="left")

# Step 3: Feature engineering
# Create aggregated transaction features for customers
transaction_features = (
    transactions_df.groupby("CustomerID")
    .agg(
        TotalSpent=("TotalValue", "sum"),
        AvgTransactionValue=("TotalValue", "mean"),
        NumTransactions=("TransactionID", "count"),
    )
    .reset_index()
)

# Merge transaction features with customer profiles
customer_features = pd.merge(customers_df, transaction_features, on="CustomerID", how="left")
customer_features.fillna(0, inplace=True)  # Fill missing values

# Selecting numeric columns for clustering
features = customer_features.select_dtypes(include=["number"])

# Step 4: Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Perform clustering
# Choose the number of clusters (between 2 and 10) and evaluate using DB Index
db_scores = []
silhouette_scores = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    
    # Calculate Davies-Bouldin Index and Silhouette Score
    db_index = davies_bouldin_score(scaled_features, labels)
    silhouette_avg = silhouette_score(scaled_features, labels)
    
    db_scores.append(db_index)
    silhouette_scores.append(silhouette_avg)

# Step 6: Choose the optimal number of clusters
optimal_clusters = cluster_range[np.argmin(db_scores)]  # Minimum DB Index
print(f"Optimal number of clusters: {optimal_clusters}")

# Final clustering with the optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(scaled_features)

# Add cluster labels to the original DataFrame
customer_features["Cluster"] = final_labels

# Step 7: Visualize clustering results
# Plot the DB Index for each number of clusters
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, db_scores, marker="o", label="DB Index")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("DB Index vs. Number of Clusters")
plt.legend()
plt.show()

# Visualize the clusters using a pair plot
sns.pairplot(customer_features, vars=["TotalSpent", "AvgTransactionValue", "NumTransactions"], hue="Cluster", palette="tab10")
plt.show()

# Step 8: Save clustering results
customer_features.to_csv(r"C:\Users\gaurgaut\Documents\CustomerClusters.csv", index=False)
print("Clustering results saved to CustomerClusters.csv")

# Step 9: Print DB Index for the optimal number of clusters
print(f"Davies-Bouldin Index for {optimal_clusters} clusters: {min(db_scores)}")

# Optional: Print silhouette scores
print(f"Silhouette Scores for {optimal_clusters} clusters: {silhouette_scores[np.argmin(db_scores)]}")
