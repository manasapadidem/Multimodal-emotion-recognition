from models.text_model import predict_text


text = "I am very happy today!"

emotion, confidence = predict_text(text)

print("Text Emotion:", emotion)
print("Text Confidence:", round(confidence, 2), "%")