import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
data = pd.read_excel(
    r"customer_segmentation.csv\Online Retail.xlsx"
)

# Display Dataset Information
print("\nDataset Information:")
print(data.info())

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove Missing Customer IDs
data = data.dropna(subset=['CustomerID'])

# Create Total Amount Column
data['TotalAmount'] = data['Quantity'] * data['UnitPrice']

# Group Data by Customer
customer_data = data.groupby('CustomerID').agg({
    'TotalAmount': 'sum'
}).reset_index()

print("\nFirst 5 Customers:")
print(customer_data.head())

print("\nNumber of Customers:")
print(customer_data.shape)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)

customer_data['Cluster'] = kmeans.fit_predict(
    customer_data[['TotalAmount']]
)

print("\nCluster Distribution:")
print(customer_data['Cluster'].value_counts())

# Show Sample Output
print("\nSample Clustered Customers:")
print(customer_data.head(10))

# Visualization
plt.scatter(
    customer_data['CustomerID'],
    customer_data['TotalAmount'],
    c=customer_data['Cluster']
)

plt.xlabel("Customer ID")
plt.ylabel("Total Amount Spent")
plt.title("Customer Segmentation using K-Means")

plt.show()