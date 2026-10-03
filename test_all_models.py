from models.face_model import predict_face
from models.speech_model import predict_speech
from models.text_model import predict_text
from models.fusion import weighted_probability_fusion


# ==========================================
# TEST TEXT
# ==========================================

text = "I am very happy today!"

text_result = predict_text(text)

print("Text Result:")
print(text_result)


# ==========================================
# TEST SPEECH
# ==========================================

audio_path = "static/uploads/test.wav"

speech_result = predict_speech(audio_path)

print("\nSpeech Result:")
print(speech_result)


# ==========================================
# TEST FACE
# ==========================================

image_path = "static/uploads/test.jpeg"

from utils.face_detection import detect_face

face = detect_face(image_path)

if face is not None:

    face_result = predict_face(face)

    print("\nFace Result:")
    print(face_result)

else:

    print("\nNo face detected.")

    # Empty probability vector if face is not detected
    face_result = {
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "neutral": 1.0,
        "sadness": 0.0,
        "surprise": 0.0
    }


# ==========================================
# EXTRACT PROBABILITY VECTORS
# ==========================================

text_probabilities = text_result[2]

speech_probabilities = speech_result[2]

face_probabilities = face_result[2] if len(face_result) == 3 else face_result


# ==========================================
# WEIGHTED PROBABILITY FUSION
# ==========================================

final_emotion, final_confidence, scores = weighted_probability_fusion(
    face_probabilities,
    speech_probabilities,
    text_probabilities
)


# ==========================================
# FINAL RESULT
# ==========================================

print("\n====================================")
print("FINAL MULTIMODAL RESULT")
print("====================================")

print("Final Emotion:", final_emotion)

print(
    "Final Confidence:",
    round(final_confidence, 2),
    "%"
)

print("\nFinal Emotion Probabilities:")

for emotion, score in scores.items():

    print(
        emotion,
        ":",
        round(score * 100, 2),
        "%"
    )