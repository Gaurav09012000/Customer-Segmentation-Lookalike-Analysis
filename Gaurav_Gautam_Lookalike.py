import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load customer and transaction data
customers_df = pd.read_csv(r"C:\Users\gaurgaut\Downloads\Customers.csv")
transactions_df = pd.read_csv(r"C:\Users\gaurgaut\Downloads\Transactions.csv")

# Step 1: Aggregate transactional data
transaction_features = transactions_df.groupby("CustomerID").agg({
    "TotalValue": "sum",  # Total spending
    "Quantity": "mean",  # Average quantity purchased
    "ProductID": "nunique"  # Product diversity (unique products purchased)
}).reset_index()

transaction_features.rename(columns={
    "TotalValue": "TotalSpending",
    "Quantity": "AvgQuantity",
    "ProductID": "ProductDiversity"
}, inplace=True)

# Step 2: Merge customer and transactional data
merged_df = pd.merge(customers_df, transaction_features, on="CustomerID", how="left")

# Fill missing transactional data with 0
merged_df[['TotalSpending', 'AvgQuantity', 'ProductDiversity']] = merged_df[
    ['TotalSpending', 'AvgQuantity', 'ProductDiversity']].fillna(0)

# Step 3: Handle missing Region values and Encode 'Region'
merged_df['Region'] = merged_df['Region'].fillna("Unknown")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Fixed argument
region_encoded = encoder.fit_transform(merged_df[['Region']])
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['Region']))
merged_df = pd.concat([merged_df, region_encoded_df], axis=1)

# Step 4: Select features for similarity calculation
features = merged_df[['TotalSpending', 'AvgQuantity', 'ProductDiversity'] + list(region_encoded_df.columns)]

# Step 5: Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 6: Calculate Cosine Similarity
similarity_matrix = cosine_similarity(scaled_features)

# Step 7: Get Lookalike Recommendations for the First 20 Customers
lookalike_dict = {}

for i in range(min(20, len(merged_df))):  # For customers C0001 to C0020 (or fewer if dataset is small)
    customer_id = merged_df['CustomerID'].iloc[i]
    similarity_scores = similarity_matrix[i]
    
    # Exclude the customer themselves and get the top 3 most similar customers
    similar_customers_idx = np.argsort(similarity_scores)[::-1][1:4]  # Exclude the first one (self)
    similar_customers = merged_df['CustomerID'].iloc[similar_customers_idx]
    scores = similarity_scores[similar_customers_idx]
    
    # Create the list of recommended customers and scores
    lookalike_dict[customer_id] = list(zip(similar_customers, scores))

# Step 8: Format and Save the Results
lookalike_list = []
for cust_id, similar_customers in lookalike_dict.items():
    for similar_customer, score in similar_customers:
        lookalike_list.append([cust_id, similar_customer, score])

lookalike_df = pd.DataFrame(lookalike_list, columns=["CustomerID", "LookalikeCustomerID", "SimilarityScore"])

# Save the recommendations to "Lookalike.csv"
lookalike_df.to_csv("Lookalike.csv", index=False)

# Display the first few rows of the result
print(lookalike_df.to_string(index=False))

# Save the DataFrame to an Excel file with column names
lookalike_df.to_excel(r"C:\Users\gaurgaut\Documents\Lookalike.xlsx", index=False)

print("Lookalike.xlsx has been saved successfully!")
