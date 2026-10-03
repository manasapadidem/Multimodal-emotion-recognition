import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================================================
# 1. Load Dataset
# =========================================================

dataset_path = "dataset/text/text_emotion_dataset.csv"

print("Loading text dataset...")

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# =========================================================
# 2. Remove Missing Values
# =========================================================

df = df.dropna(subset=["text", "emotion"])

df["text"] = df["text"].astype(str)
df["emotion"] = df["emotion"].astype(str)

texts = df["text"].values
labels = df["emotion"].values


# =========================================================
# 3. Encode Emotion Labels
# =========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

print("\nEmotion Classes:")
print(label_encoder.classes_)


# =========================================================
# 4. Tokenization
# =========================================================

MAX_WORDS = 20000
MAX_LEN = 100

print("\nTokenizing text...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("Text converted into sequences.")
print("Feature Shape:", X.shape)


# =========================================================
# 5. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# =========================================================
# 6. Calculate Class Weights
# =========================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {}

for i in range(len(classes)):
    class_weights[int(classes[i])] = float(class_weights_array[i])

print("\nClass Weights:")
print(class_weights)


# =========================================================
# 7. Create Bidirectional LSTM Model
# =========================================================

model = Sequential()

model.add(
    Input(shape=(MAX_LEN,))
)

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    )
)

model.add(
    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    )
)

model.add(
    Dropout(0.4)
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(
    Dropout(0.4)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)


# =========================================================
# 8. Compile Model
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 9. Model Summary
# =========================================================

print("\nLSTM Model Summary:")

model.summary()


# =========================================================
# 10. Callbacks
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# =========================================================
# 11. Train Model
# =========================================================

print("\n====================================")
print("Training Text LSTM Model...")
print("====================================")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=30,

    batch_size=64,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# =========================================================
# 12. Evaluate Model
# =========================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n====================================")
print("Text LSTM Accuracy:", accuracy * 100)
print("====================================")


# =========================================================
# 13. Create Model Folder
# =========================================================

os.makedirs(
    "model",
    exist_ok=True
)


# =========================================================
# 14. Save LSTM Model
# =========================================================

model.save(
    "model/text_lstm_model.h5"
)


# =========================================================
# 15. Save Tokenizer
# =========================================================

with open(
    "model/text_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


# =========================================================
# 16. Save Emotion Labels
# =========================================================

np.save(
    "model/text_labels.npy",
    label_encoder.classes_
)


# =========================================================
# 17. Final Message
# =========================================================

print("\n====================================")
print("Text LSTM Model Saved Successfully!")
print("====================================")

print("Model     : model/text_lstm_model.h5")
print("Tokenizer : model/text_tokenizer.pkl")
print("Labels    : model/text_labels.npy")