import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_rows = 150

data = {
    "passenger_id": [f"ID_{i:03d}" for i in range(1, n_rows + 1)],
    "age": np.random.normal(35, 12, n_rows).round(1),
    "salary": [f"${np.random.randint(40, 150) * 1000:,}" for _ in range(n_rows)],
    "gender": np.random.choice(["Male", "Female", "Other"], size=n_rows, p=[0.48, 0.48, 0.04]),
    "city": ["New York"] * n_rows, # Constant column
    "purchased": np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Inject missing values
# Age: 12 missing values
df.loc[np.random.choice(n_rows, 12, replace=False), "age"] = np.nan
# Gender: 8 missing values
df.loc[np.random.choice(n_rows, 8, replace=False), "gender"] = np.nan

# Inject outliers in Age column (extreme values)
df.loc[10, "age"] = 120.0
df.loc[50, "age"] = -5.0
df.loc[100, "age"] = 105.0

# Inject duplicate rows
duplicates = df.iloc[[12, 45, 78, 120]].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# Save to CSV
output_path = "sample_data.csv"
df.to_csv(output_path, index=False)
print(f"Sample dataset generated successfully at '{output_path}'.")
print(f"Shape: {df.shape}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Missing Values:\n{df.isnull().sum()}")
