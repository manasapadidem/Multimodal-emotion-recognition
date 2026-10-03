import pandas as pd
import os

# ==========================================
# Paths
# ==========================================

train_path = "dataset/goemotions/train.tsv"
dev_path = "dataset/goemotions/dev.tsv"
test_path = "dataset/goemotions/test.tsv"

emotion_path = "dataset/goemotions/emotions.txt"

output_path = "dataset/text/text_emotion_dataset.csv"


# ==========================================
# Load emotion names
# ==========================================

with open(emotion_path, "r", encoding="utf-8") as file:
    emotions = [line.strip() for line in file.readlines()]

print("Total emotions in GoEmotions:", len(emotions))


# ==========================================
# Load datasets
# ==========================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    train_path,
    sep="\t",
    header=None
)

dev_df = pd.read_csv(
    dev_path,
    sep="\t",
    header=None
)

test_df = pd.read_csv(
    test_path,
    sep="\t",
    header=None
)


# ==========================================
# Combine datasets
# ==========================================

df = pd.concat(
    [train_df, dev_df, test_df],
    ignore_index=True
)

print("Total rows:", len(df))


# ==========================================
# Rename columns
# ==========================================

df.columns = [
    "text",
    "emotion_id",
    "comment_id"
]


# ==========================================
# Convert emotion ID to emotion name
# ==========================================

df["emotion_id"] = df["emotion_id"].astype(str)

def get_emotion(emotion_id):

    # GoEmotions can contain multiple labels.
    # We only keep rows having one emotion.

    if "," in emotion_id:
        return None

    try:
        index = int(emotion_id)

        if index < len(emotions):
            return emotions[index]

    except:
        return None

    return None


df["emotion"] = df["emotion_id"].apply(get_emotion)


# ==========================================
# Seven emotion mapping
# ==========================================

emotion_mapping = {

    # Anger group
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",

    # Disgust
    "disgust": "disgust",

    # Fear group
    "fear": "fear",

    # Joy group
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",

    # Sadness group
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",

    # Surprise
    "surprise": "surprise",

    # Neutral
    "neutral": "neutral"
}


# ==========================================
# Apply mapping
# ==========================================

df["emotion"] = df["emotion"].map(emotion_mapping)


# ==========================================
# Remove unsupported emotions
# ==========================================

df = df.dropna(
    subset=["emotion"]
)


# ==========================================
# Keep only required columns
# ==========================================

df = df[
    ["text", "emotion"]
]


# ==========================================
# Remove empty text
# ==========================================

df["text"] = df["text"].astype(str)

df = df[
    df["text"].str.strip() != ""
]


# ==========================================
# Remove duplicate text
# ==========================================

df = df.drop_duplicates(
    subset=["text"]
)


# ==========================================
# Create text folder
# ==========================================

os.makedirs(
    "dataset/text",
    exist_ok=True
)


# ==========================================
# Save dataset
# ==========================================

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# Display results
# ==========================================

print("\n====================================")
print("Text dataset created successfully!")
print("====================================")

print("\nTotal samples:", len(df))

print("\nEmotion distribution:")
print(
    df["emotion"].value_counts()
)

print("\nEmotion percentages:")
print(
    df["emotion"].value_counts(normalize=True) * 100
)

print("\nFirst 10 rows:")
print(
    df.head(10).to_string(index=False)
)

print("\nSaved file:")
print(output_path)