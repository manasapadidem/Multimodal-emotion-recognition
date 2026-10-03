import pandas as pd

dataset_path = "dataset/text/goemotions_ekman_singlelabel.csv"

df = pd.read_csv(dataset_path)

print("Emotion Distribution:")
print(df["ekman_emotion"].value_counts())

print("\nEmotion Percentages:")
print(df["ekman_emotion"].value_counts(normalize=True) * 100)

print("\nSample Text and Labels:")
print(df[["text", "ekman_emotion"]].head(20).to_string(index=False))