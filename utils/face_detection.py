import cv2
import os

# Load Haar Cascade
cascade_path = os.path.join(
    "haarcascades",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_face(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # No face detected
    if len(faces) == 0:
        return None

    # Get the first detected face
    x, y, w, h = faces[0]

    # Crop the face
    face = image[y:y+h, x:x+w]

    return face