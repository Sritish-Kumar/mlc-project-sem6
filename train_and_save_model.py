from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocess import CLASS_ORDER, clean_text, ensure_nltk_data

DATA_PATH = Path("data/cyberbullying_tweets.csv")
MODEL_DIR = Path("models")
RANDOM_STATE = 42


def build_artifacts() -> None:
    ensure_nltk_data()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()
    df["cleaned_text"] = df["tweet_text"].apply(clean_text)

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["cyberbullying_type"])

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label_encoded"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_encoded"],
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(x_train_raw)
    x_test = vectorizer.transform(x_test_raw)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    label_order = label_encoder.transform(CLASS_ORDER)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    evaluation_data = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=label_order,
        ),
        "label_order": label_order,
        "classes": label_encoder.classes_,
        "class_order": CLASS_ORDER,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }

    joblib.dump(model, MODEL_DIR / "logistic_regression_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(evaluation_data, MODEL_DIR / "evaluation_data.pkl")

    print("Saved model artifacts to models/")
    print(f"Logistic Regression accuracy: {evaluation_data['accuracy']:.6f}")


if __name__ == "__main__":
    build_artifacts()
