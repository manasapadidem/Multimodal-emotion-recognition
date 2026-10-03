import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ==========================================
# Load Face CNN Model
# ==========================================

model = load_model("model/emotion_model.h5")


# ==========================================
# Emotion Labels
# IMPORTANT:
# Order must match the CNN training order
# ==========================================

emotion_labels = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]


# ==========================================
# Predict Face Emotion Probabilities
# ==========================================

def predict_face(face):

    # Convert image to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Resize image
    gray = cv2.resize(gray, (48, 48))

    # Normalize pixels
    gray = gray.astype("float32") / 255.0

    # Reshape for CNN
    gray = gray.reshape(1, 48, 48, 1)

    # Get probability vector
    prediction = model.predict(
        gray,
        verbose=0
    )[0]

    # Get highest probability
    index = np.argmax(prediction)

    emotion = emotion_labels[index]

    confidence = float(
        prediction[index] * 100
    )

    # Create probability dictionary
    probabilities = {}

    for i in range(len(emotion_labels)):
        probabilities[emotion_labels[i]] = float(
            prediction[i]
        )

    return emotion, confidence, probabilities