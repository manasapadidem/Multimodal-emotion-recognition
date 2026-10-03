import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

print("TensorFlow Version:", tf.__version__)

# --------------------------------
# Dataset path
# --------------------------------
dataset_path = "speech_dataset"

features = []
labels = []

# Maximum number of MFCC time frames
MAX_LEN = 174

print("Loading Speech Dataset...")

# --------------------------------
# Emotion mapping
# RAVDESS emotion codes
# --------------------------------
emotion_map = {
    "01": "Neutral",
    "02": "Neutral",    # Calm mapped to Neutral
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

# --------------------------------
# Load audio files
# --------------------------------
for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            # Get emotion code from filename
            parts = file.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                continue

            emotion = emotion_map[emotion_code]

            try:

                # Load audio
                signal, sample_rate = librosa.load(
                    file_path,
                    sr=22050
                )

                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=signal,
                    sr=sample_rate,
                    n_mfcc=40
                )

                # Transpose:
                # (40, time) -> (time, 40)
                mfcc = mfcc.T

                # Pad or truncate
                if mfcc.shape[0] < MAX_LEN:

                    padding = MAX_LEN - mfcc.shape[0]

                    mfcc = np.pad(
                        mfcc,
                        ((0, padding), (0, 0)),
                        mode="constant"
                    )

                else:

                    mfcc = mfcc[:MAX_LEN, :]

                features.append(mfcc)
                labels.append(emotion)

            except Exception as e:
                print("Error:", file_path)
                print(e)


print("Dataset Loaded Successfully!")

# --------------------------------
# Convert to NumPy arrays
# --------------------------------
X = np.array(features)
y = np.array(labels)

print("Feature Shape:", X.shape)
print("Labels Shape:", y.shape)

# --------------------------------
# Encode labels
# --------------------------------
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Emotion Classes:")
print(label_encoder.classes_)

# --------------------------------
# Convert labels to categorical
# --------------------------------
y_categorical = to_categorical(
    y_encoded,
    num_classes=len(label_encoder.classes_)
)

# --------------------------------
# Train/Test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------
# LSTM Model
# --------------------------------
model = Sequential()

model.add(
    Masking(
        mask_value=0.0,
        input_shape=(MAX_LEN, 40)
    )
)

model.add(
    LSTM(
        128,
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        64
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)

# --------------------------------
# Compile
# --------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------
# Display model
# --------------------------------
model.summary()

# --------------------------------
# Train
# --------------------------------
print("Training LSTM Model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# --------------------------------
# Evaluate
# --------------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("Speech LSTM Accuracy:", accuracy * 100)

# --------------------------------
# Create model folder
# --------------------------------
os.makedirs("model", exist_ok=True)

# --------------------------------
# Save LSTM model
# --------------------------------
model.save(
    "model/speech_lstm_model.h5"
)

# Save label encoder classes
np.save(
    "model/speech_labels.npy",
    label_encoder.classes_
)

print()
print("Speech LSTM Model Saved Successfully!")
print("Model: model/speech_lstm_model.h5")
print("Labels: model/speech_labels.npy")