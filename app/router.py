from flask import request, jsonify
from app import app
from src.models.predict_model import load_model, predict_email
from src.data.preprocessor import preprocess_text, detect_language
from config import MODEL_PATH, VECTORIZER_PATH

model = load_model(MODEL_PATH)
vectorizer = load_model(VECTORIZER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data['email']
    prediction = predict_email(model, vectorizer, email_text)
    return jsonify({'prediction': 'spam' if prediction == 1 else 'not spam'})
