from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch


def create_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer, vectorizer.fit_transform(texts)


def get_language_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def combine_features(tfidf_features, language_embeddings):
    return torch.cat([torch.tensor(tfidf_features.toarray()), language_embeddings], dim=1)
