import pandas as pd

dataset_path = "dataset/text/goemotions_ekman_singlelabel.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")

print("\nColumns:")
print(df.columns)

print("\nNumber of rows:")
print(len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nEmotion classes:")
print(df["ekman_emotion"].unique())