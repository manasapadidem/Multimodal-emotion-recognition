# Common emotion order
EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]


# Model weights
FACE_WEIGHT = 0.30
SPEECH_WEIGHT = 0.30
TEXT_WEIGHT = 0.40


def normalize_emotion(emotion):

    emotion = str(emotion).lower().strip()

    mapping = {
        "angry": "anger",
        "anger": "anger",

        "disgust": "disgust",

        "fear": "fear",

        "happy": "joy",
        "joy": "joy",

        "neutral": "neutral",

        "sad": "sadness",
        "sadness": "sadness",

        "surprise": "surprise"
    }

    return mapping.get(emotion, emotion)


def normalize_probabilities(probabilities):

    result = {}

    for emotion in EMOTIONS:
        result[emotion] = 0.0

    # Convert every model's labels
    # into the common emotion names
    for label, probability in probabilities.items():

        normalized_label = normalize_emotion(label)

        if normalized_label in result:
            result[normalized_label] += float(probability)

    # Make sure probabilities add up to 1
    total = sum(result.values())

    if total > 0:

        for emotion in EMOTIONS:
            result[emotion] = result[emotion] / total

    return result


def weighted_probability_fusion(
        face_probabilities,
        speech_probabilities,
        text_probabilities):

    # Normalize each model's probability vector
    face = normalize_probabilities(
        face_probabilities
    )

    speech = normalize_probabilities(
        speech_probabilities
    )

    text = normalize_probabilities(
        text_probabilities
    )

    # Final weighted probability
    scores = {}

    for emotion in EMOTIONS:

        scores[emotion] = (
            FACE_WEIGHT * face[emotion]
            + SPEECH_WEIGHT * speech[emotion]
            + TEXT_WEIGHT * text[emotion]
        )

    # Find final emotion
    final_emotion = max(
        scores,
        key=scores.get
    )

    # Convert final probability to percentage
    final_confidence = (
        scores[final_emotion] * 100
    )

    return (
        final_emotion,
        final_confidence,
        scores
    )