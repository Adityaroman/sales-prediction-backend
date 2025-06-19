import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'Customer_ID': range(10000),
    'Age': np.random.randint(18, 70, 10000),
    'Gender': np.random.choice(['M', 'F'], 10000),
    'Marital_Status': np.random.choice(['Married', 'Single'], 10000),
    'State': np.random.choice(['Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Haryana', 'Delhi'], 10000),
    'Product_Category': np.random.choice(['Food', 'Clothing', 'Electronics', 'Home'], 10000),
    'Order_Date': pd.date_range(start='2023-01-01', end='2025-06-17', periods=10000).strftime('%Y-%m-%d'),
    'Orders': np.random.randint(1, 10, 10000),
    'Amount': np.random.randint(500, 50000, 10000)
}
df = pd.DataFrame(data)
df.to_csv('data/sales_data.csv', index=False)
print("Dataset generated and saved to data/sales_data.csv")