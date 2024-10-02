import re
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


def preprocess_text(text, language):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    if language == 'en':
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in stop_words])
    elif language == 'vi':
        # Xử lý tiếng Việt
        text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
        # ... (thêm các quy tắc khác)

    return text


def preprocess_dataset(df):
    df['processed_text'] = df.apply(lambda row: preprocess_text(row['text'], row['language']), axis=1)
    return df
