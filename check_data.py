import pandas as pd

# Load dataset
df = pd.read_csv("data/synthetic_timeseries.csv")

print("\n--- DATA PREVIEW ---")
print(df.head())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- MISSING VALUES ---")
print(df.isna().sum())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())
