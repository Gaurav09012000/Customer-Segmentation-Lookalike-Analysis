# Importing necessary libraries
import pandas as pd
import numpy as np

# Paths to your datasets
customers_path = r"C:\Users\gaurgaut\Downloads\Customers.csv"
products_path = r"C:\Users\gaurgaut\Downloads\Products.csv"
transactions_path = r"C:\Users\gaurgaut\Downloads\Transactions.csv"

# Loading the datasets
try:
    customers_df = pd.read_csv(customers_path)
    print("Customers dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Customers dataset: {e}")

try:
    products_df = pd.read_csv(products_path)
    print("Products dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Products dataset: {e}")

try:
    transactions_df = pd.read_csv(transactions_path)
    print("Transactions dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Transactions dataset: {e}")

# Cleaning the datasets
def clean_data(df, name):
    print(f"\nCleaning {name} dataset...")

    # Checking for missing values
    print(f"Missing values before cleaning in {name}:\n{df.isnull().sum()}\n")

    # Dropping rows with all NaN values
    df.dropna(how='all', inplace=True)

    # Filling missing values with appropriate methods (example: mean, mode, or 'Unknown')
    df.fillna({'Column1': 'Unknown', 'Column2': df['Column2'].mode()[0] if 'Column2' in df.columns else np.nan}, inplace=True)

    # Dropping duplicate rows
    before_duplicates = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_duplicates = df.shape[0]
    print(f"Removed {before_duplicates - after_duplicates} duplicate rows in {name}.")

    # Checking data types
    print(f"Data types before conversion in {name}:\n{df.dtypes}\n")

    # Converting columns to appropriate data types (example shown for integers and datetime)
    for column in df.select_dtypes(include=['object']).columns:
        if 'date' in column.lower():
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif df[column].str.isdigit().all():
            df[column] = pd.to_numeric(df[column], errors='coerce')

    print(f"Data types after conversion in {name}:\n{df.dtypes}\n")

    print(f"{name} dataset cleaned successfully!")
    return df

# Cleaning each dataset
customers_df = clean_data(customers_df, "Customers")
products_df = clean_data(products_df, "Products")
transactions_df = clean_data(transactions_df, "Transactions")

# Checking cleaned datasets
print("\nCleaned Customers Dataset:")
print(customers_df.head(), "\n")

print("\nCleaned Products Dataset:")
print(products_df.head(), "\n")

print("\nCleaned Transactions Dataset:")
print(transactions_df.head(), "\n")

# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize column names
transactions_df.columns = transactions_df.columns.str.strip()  # Remove leading/trailing spaces
transactions_df.columns = transactions_df.columns.str.lower()  # Convert column names to lowercase

# Analyzing Key Metrics with Visualizations
print("\n--- Analysis of Key Metrics ---")

# 1. Customer Behavior
print("\n1. Customer Behavior:")
if 'customerid' in transactions_df.columns:
    customer_purchase_counts = transactions_df['customerid'].value_counts()
    print(f"Top 5 Customers by Number of Purchases:\n{customer_purchase_counts.head()}\n")

    # Visualization: Top 5 Customers by Purchases
    plt.figure(figsize=(8, 6))
    customer_purchase_counts.head(5).plot(kind='bar', color='skyblue')
    plt.title('Top 5 Customers by Number of Purchases')
    plt.xlabel('Customer ID')
    plt.ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Error: 'CustomerID' column is missing in the transactions dataset.")

# 2. Product Popularity
print("\n2. Product Popularity:")
if 'productid' in transactions_df.columns:
    product_popularity = transactions_df['productid'].value_counts()
    print(f"Top 5 Most Popular Products:\n{product_popularity.head()}\n")

    # Visualization: Top 5 Most Popular Products
    plt.figure(figsize=(8, 6))
    product_popularity.head(5).plot(kind='bar', color='orange')
    plt.title('Top 5 Most Popular Products')
    plt.xlabel('Product ID')
    plt.ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Error: 'ProductID' column is missing in the transactions dataset.")

# 3. Revenue Trends
print("\n3. Revenue Trends:")
if 'price' in transactions_df.columns and 'quantity' in transactions_df.columns:
    transactions_df['revenue'] = transactions_df['price'] * transactions_df['quantity']
    total_revenue = transactions_df['revenue'].sum()
    print(f"Total Revenue: {total_revenue}\n")
    
    revenue_by_product = transactions_df.groupby('productid')['revenue'].sum().sort_values(ascending=False)
    print(f"Top 5 Products by Revenue:\n{revenue_by_product.head()}\n")

    # Visualization: Top 5 Products by Revenue
    plt.figure(figsize=(8, 6))
    revenue_by_product.head(5).plot(kind='bar', color='green')
    plt.title('Top 5 Products by Revenue')
    plt.xlabel('Product ID')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Error: 'Price' or 'Quantity' column is missing in the transactions dataset.")

# 4. Monthly Revenue Trends
if 'transactiondate' in transactions_df.columns:
    transactions_df['transactiondate'] = pd.to_datetime(transactions_df['transactiondate'], errors='coerce')
    transactions_df['month'] = transactions_df['transactiondate'].dt.to_period('M')
    monthly_revenue = transactions_df.groupby('month')['revenue'].sum()
    print(f"Monthly Revenue Trends:\n{monthly_revenue}\n")

    # Visualization: Monthly Revenue Trends
    plt.figure(figsize=(10, 6))
    monthly_revenue.plot(kind='line', marker='o', color='purple')
    plt.title('Monthly Revenue Trends')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
else:
    print("Error: 'TransactionDate' column is missing in the transactions dataset.")
