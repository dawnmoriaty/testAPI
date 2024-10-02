import joblib

from src.data.preprocessor import detect_language, preprocess_text


def load_model(path):
    return joblib.load(path)

def predict_email(model, vectorizer, email_text):
    processed_text = preprocess_text(email_text, detect_language(email_text))
    features = vectorizer.transform([processed_text])
    return model.predict(features)[0]
