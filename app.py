from flask import Flask, render_template, request

import os

from utils.face_detection import detect_face
from models.face_model import predict_face
from models.speech_model import predict_speech
from models.text_model import predict_text
from models.fusion import weighted_probability_fusion


app = Flask(__name__)


# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# OTHER PAGES
# ==========================================

@app.route("/multimodal")
def multimodal():

    return render_template(
        "multimodal.html"
    )


@app.route("/face")
def face():

    return render_template(
        "face.html"
    )


@app.route("/speech")
def speech():

    return render_template(
        "speech.html"
    )


@app.route("/text")
def text():

    return render_template(
        "text.html"
    )


@app.route("/combine")
def combine():

    return render_template(
        "combine.html"
    )


@app.route("/result")
def result():

    return render_template(
        "result.html"
    )


# ==========================================
# FACE PREDICTION
# ==========================================

@app.route(
    "/predict_face",
    methods=["POST"]
)
def predict_face_route():

    image = request.files["image"]

    if image.filename == "":
        return "No image selected."

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    # Detect face
    face_image = detect_face(
        image_path
    )

    if face_image is None:
        return "No face detected."

    # Predict face emotion
    emotion, confidence, probabilities = predict_face(
        face_image
    )

    return render_template(
        "result.html",
        emotion=emotion,
        confidence=round(
            confidence,
            2
        ),
        probabilities=probabilities,
        filename=image.filename
    )


# ==========================================
# SPEECH PREDICTION
# ==========================================

@app.route(
    "/predict_speech",
    methods=["POST"]
)
def predict_speech_route():

    audio = request.files["audio"]

    if audio.filename == "":
        return "No audio selected."

    audio_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        audio.filename
    )

    audio.save(audio_path)

    # Predict speech emotion
    emotion, confidence, probabilities = predict_speech(
        audio_path
    )

    return render_template(
        "result.html",
        emotion=emotion,
        confidence=round(
            confidence,
            2
        ),
        probabilities=probabilities,
        filename=None
    )


# ==========================================
# TEXT PREDICTION
# ==========================================

@app.route(
    "/predict_text",
    methods=["POST"]
)
def predict_text_route():

    text_input = request.form["text"]

    if text_input.strip() == "":
        return "Please enter some text."

    # Predict text emotion
    emotion, confidence, probabilities = predict_text(
        text_input
    )

    return render_template(
        "result.html",
        emotion=emotion,
        confidence=round(
            confidence,
            2
        ),
        probabilities=probabilities,
        filename=None
    )


# ==========================================
# MULTIMODAL PREDICTION
# ==========================================

@app.route(
    "/predict_multimodal",
    methods=["POST"]
)
def predict_multimodal():

    # ======================================
    # TEXT PREDICTION
    # ======================================

    text_input = request.form.get(
        "text",
        ""
    )

    if text_input.strip() == "":
        return "Please enter text."

    text_result = predict_text(
        text_input
    )

    # Text result
    text_emotion = text_result[0]
    text_confidence = text_result[1]
    text_probabilities = text_result[2]


    # ======================================
    # SPEECH PREDICTION
    # ======================================

    audio = request.files.get(
        "audio"
    )

    if audio is None or audio.filename == "":
        return "Please upload an audio file."

    audio_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        audio.filename
    )

    audio.save(audio_path)

    speech_result = predict_speech(
        audio_path
    )

    # Speech result
    speech_emotion = speech_result[0]
    speech_confidence = speech_result[1]
    speech_probabilities = speech_result[2]


    # ======================================
    # FACE PREDICTION
    # ======================================

    image = request.files.get(
        "image"
    )

    if image is None or image.filename == "":
        return "Please upload an image."

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    # Detect face
    face_image = detect_face(
        image_path
    )

    if face_image is None:
        return "No face detected in the uploaded image."

    face_result = predict_face(
        face_image
    )

    # Face result
    face_emotion = face_result[0]
    face_confidence = face_result[1]
    face_probabilities = face_result[2]


    # ======================================
    # WEIGHTED PROBABILITY FUSION
    # ======================================

    final_emotion, final_confidence, scores = weighted_probability_fusion(
        face_probabilities,
        speech_probabilities,
        text_probabilities
    )


    # ======================================
    # FINAL RESULT PAGE
    # ======================================

    return render_template(
        "result.html",

        # -------------------------------
        # FACE RESULT
        # -------------------------------

        face_emotion=face_emotion,
        face_confidence=round(
            face_confidence,
            2
        ),

        # -------------------------------
        # SPEECH RESULT
        # -------------------------------

        speech_emotion=speech_emotion,
        speech_confidence=round(
            speech_confidence,
            2
        ),

        # -------------------------------
        # TEXT RESULT
        # -------------------------------

        text_emotion=text_emotion,
        text_confidence=round(
            text_confidence,
            2
        ),

        # -------------------------------
        # FUSION WEIGHTS
        # -------------------------------

        face_weight=30,
        speech_weight=30,
        text_weight=40,

        # -------------------------------
        # FINAL RESULT
        # -------------------------------

        emotion=final_emotion,

        confidence=round(
            final_confidence,
            2
        ),

        # -------------------------------
        # FINAL PROBABILITIES
        # -------------------------------

        probabilities=scores,

        # -------------------------------
        # IMAGE
        # -------------------------------

        filename=image.filename
    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )