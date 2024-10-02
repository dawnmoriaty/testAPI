import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')
#
# ENGLISH_DATA_PATH = os.path.join(RAW_DATA_DIR, 'english_emails.csv')
# VIETNAMESE_DATA_PATH = os.path.join(RAW_DATA_DIR, 'vietnamese_emails.csv')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'spam_classifier.joblib')
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.joblib')
