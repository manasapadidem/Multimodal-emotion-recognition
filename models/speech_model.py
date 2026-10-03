import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model("model/speech_lstm_model.h5")

# Load emotion labels
emotion_labels = np.load(
    "model/speech_labels.npy",
    allow_pickle=True
)

MAX_LEN = 174


def predict_speech(audio_path):

    # Load audio
    signal, sample_rate = librosa.load(
        audio_path,
        sr=22050
    )

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    # Convert:
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

    # Add batch dimension
    mfcc = mfcc.reshape(1, MAX_LEN, 40)

    # Predict
    prediction = model.predict(
        mfcc,
        verbose=0
    )[0]

    # Highest probability
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