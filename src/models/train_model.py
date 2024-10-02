from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def save_model(model, path):
    joblib.dump(model, path)

def train_and_save_model(X, y, model_path):
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    print(evaluate_model(model, X_test, y_test))
    save_model(model, model_path)
