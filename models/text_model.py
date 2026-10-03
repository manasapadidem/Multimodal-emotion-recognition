import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained BiLSTM model
model = load_model("model/text_lstm_model.h5")

# Load tokenizer
with open("model/text_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load emotion labels
emotion_labels = np.load(
    "model/text_labels.npy",
    allow_pickle=True
)

MAX_LEN = 100


def predict_text(text):

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Find highest probability emotion
    index = np.argmax(prediction)

    emotion = str(emotion_labels[index])

    confidence = float(
        prediction[index] * 100
    )

    # Store probability of every emotion
    probabilities = {}

    for i in range(len(emotion_labels)):
        probabilities[str(emotion_labels[i])] = float(prediction[i])

    return emotion, confidence, probabilities