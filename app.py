from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from preprocess import CLASS_ORDER, DISPLAY_NAMES, clean_text, ensure_nltk_data


DATA_PATH = Path("data/cyberbullying_tweets.csv")
MODEL_DIR = Path("models")

MODEL_SCORES = {
    "Naive Bayes": 0.752280113219415,
    "Passive Aggressive": 0.7848831114372575,
    "Random Forest": 0.799769367858266,
    "SVM": 0.8059545025684034,
    "Logistic Regression": 0.8099381486528986,
}

PAGE_OPTIONS = [
    "Tweet Classifier",
    "Dataset Overview",
    "Model Performance",
    "Processing Pipeline",
    "About Project",
]


st.set_page_config(
    page_title="Cyberbullying Tweet Classifier",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .prediction-box {
        border-left: 6px solid #0B3D91;
        background: #f8fafc;
        padding: 1rem 1.2rem;
        border-radius: 8px;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def setup_nltk() -> None:
    ensure_nltk_data()


@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "logistic_regression_model.pkl"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    missing = [
        str(path)
        for path in [model_path, vectorizer_path, encoder_path]
        if not path.exists()
    ]
    if missing:
        return None, None, None, missing

    return (
        joblib.load(model_path),
        joblib.load(vectorizer_path),
        joblib.load(encoder_path),
        [],
    )


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_evaluation_data():
    path = MODEL_DIR / "evaluation_data.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def readable_label(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='small-muted'>{subtitle}</p>", unsafe_allow_html=True)


def probability_chart(probabilities: np.ndarray, classes: np.ndarray) -> None:
    data = pd.DataFrame(
        {
            "Class": [readable_label(label) for label in classes],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=True)

    colors = [
        "#0B3D91" if value == data["Probability"].max() else "#B8C2CC"
        for value in data["Probability"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(data["Class"], data["Probability"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class Probability Distribution")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for index, value in enumerate(data["Probability"]):
        ax.text(value + 0.01, index, f"{value:.1%}", va="center", fontweight="bold")

    st.pyplot(fig, use_container_width=True)


def classifier_page(model, vectorizer, label_encoder) -> None:
    render_header(
        "Tweet Classifier",
        "Classify one tweet at a time with the saved Logistic Regression model.",
    )

    tweet = st.text_area(
        "Enter a tweet",
        height=150,
        placeholder="Paste or type a tweet here...",
    )

    show_cleaned = st.checkbox("Show cleaned text", value=True)
    show_probabilities = st.checkbox("Show class probabilities", value=True)

    if st.button("Classify Tweet", type="primary"):
        if not tweet.strip():
            st.warning("Please enter a tweet before classifying.")
            return

        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning("The tweet becomes empty after preprocessing. Try adding more text.")
            return

        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        predicted_display = readable_label(predicted_label)

        probabilities = model.predict_proba(vectorized)[0]
        probability_index = int(np.where(model.classes_ == prediction)[0][0])
        confidence = float(probabilities[probability_index])

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="small-muted">Predicted class</div>
                <h2>{predicted_display}</h2>
                <strong>Confidence:</strong> {confidence:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_cleaned:
            st.subheader("Cleaned Text")
            st.code(cleaned)

        if show_probabilities:
            probability_chart(probabilities, label_encoder.classes_)


def dataset_overview_page(df: pd.DataFrame) -> None:
    render_header(
        "Dataset Overview",
        "Explore the class balance and inspect real examples from the tweet dataset.",
    )

    class_counts = df["cyberbullying_type"].value_counts()
    total_tweets = int(class_counts.sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tweets", f"{total_tweets:,}")
    col2.metric("Classes", f"{class_counts.shape[0]}")
    col3.metric("Largest Class", readable_label(class_counts.idxmax()))

    chart_data = (
        class_counts.rename_axis("Class")
        .reset_index(name="Tweets")
        .assign(Display=lambda frame: frame["Class"].map(readable_label))
        .sort_values("Tweets", ascending=True)
    )

    left, right = st.columns([1.15, 1])
    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(chart_data["Display"], chart_data["Tweets"], color="#0B3D91")
        ax.set_xlabel("Tweet Count")
        ax.set_title("Class Distribution")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Class Counts")
        table = chart_data.sort_values("Tweets", ascending=False)[["Display", "Tweets"]]
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Sample Tweets")
    selected_display = st.selectbox(
        "Choose a class",
        [readable_label(label) for label in CLASS_ORDER],
    )
    selected_label = next(
        label for label in CLASS_ORDER if readable_label(label) == selected_display
    )
    sample_count = st.slider("Number of examples", 1, 10, 5)
    samples = (
        df[df["cyberbullying_type"] == selected_label]["tweet_text"]
        .sample(min(sample_count, class_counts[selected_label]), random_state=42)
        .reset_index(drop=True)
    )
    st.dataframe(pd.DataFrame({"Tweet": samples}), hide_index=True, use_container_width=True)


def model_performance_page(evaluation_data, label_encoder) -> None:
    render_header(
        "Model Performance",
        "Compare model accuracy and inspect the Logistic Regression model in detail.",
    )

    scores = (
        pd.DataFrame(
            {"Model": list(MODEL_SCORES.keys()), "Accuracy": list(MODEL_SCORES.values())}
        )
        .sort_values("Accuracy", ascending=True)
        .reset_index(drop=True)
    )
    colors = [
        "#0B3D91" if model == "Logistic Regression" else "#B8C2CC"
        for model in scores["Model"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(scores["Model"], scores["Accuracy"], color=colors)
    ax.set_xlim(0.70, 0.83)
    ax.set_xlabel("Accuracy Score")
    ax.set_title("Performance Leaderboard")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, score in zip(bars, scores["Accuracy"]):
        ax.text(
            score + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontweight="bold",
        )
    st.pyplot(fig, use_container_width=True)

    if evaluation_data is None:
        st.info(
            "Detailed Logistic Regression metrics will appear after running "
            "`python train_and_save_model.py`."
        )
        return

    class_labels = [readable_label(label) for label in CLASS_ORDER]
    st.subheader("Logistic Regression Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        evaluation_data["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Precision, Recall, and F1-Score by Class")
    report = evaluation_data.get("classification_report")
    if report is None:
        st.warning("Classification report data is missing from evaluation artifacts.")
        return

    metrics = pd.DataFrame(
        {
            "Class": [readable_label(label) for label in CLASS_ORDER],
            "Precision": [report[label]["precision"] for label in CLASS_ORDER],
            "Recall": [report[label]["recall"] for label in CLASS_ORDER],
            "F1-Score": [report[label]["f1-score"] for label in CLASS_ORDER],
        }
    )

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width, metrics["Precision"], width=width, label="Precision", color="#0B3D91")
    ax.bar(x, metrics["Recall"], width=width, label="Recall", color="#2A9D8F")
    ax.bar(x + width, metrics["F1-Score"], width=width, label="F1-Score", color="#E9C46A")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Logistic Regression Class-Level Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Class"], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Classification Report")
    st.dataframe(metrics, hide_index=True, use_container_width=True)


def processing_pipeline_page() -> None:
    render_header(
        "Processing Pipeline",
        "See exactly how raw tweet text becomes model-ready input.",
    )

    st.markdown(
        """
        1. Lowercase text
        2. Remove URLs, mentions, hashtags, and punctuation
        3. Strip extra whitespace
        4. Remove English stopwords
        5. Lemmatize words
        6. Convert to TF-IDF features
        7. Predict with Logistic Regression
        """
    )

    demo_text = st.text_area(
        "Try the preprocessing step",
        value="I hate this!!! Visit https://example.com @user #angry",
        height=120,
    )
    st.subheader("Cleaned Output")
    st.code(clean_text(demo_text))


def about_page() -> None:
    render_header(
        "About Project",
        "A compact machine learning project for cyberbullying tweet classification.",
    )

    st.markdown(
        """
        This app classifies tweets into six categories: Age, Ethnicity, Gender,
        Religion, Other Cyberbullying, and Not Cyberbullying. The notebook compared
        five machine learning models and selected Logistic Regression as the live
        prediction model because it achieved the highest accuracy.

        The app is intended as a project demonstration and decision-support tool.
        It should not be used as the only moderation decision maker. Sarcasm,
        vague language, slang, and context outside the tweet can still affect
        prediction quality.
        """
    )

    st.subheader("Production Model")
    st.write("Logistic Regression with TF-IDF features.")
    st.subheader("Why Other Models Appear")
    st.write(
        "Naive Bayes, Random Forest, Passive Aggressive, and SVM are shown in "
        "comparison charts to explain why Logistic Regression was selected."
    )


def main() -> None:
    setup_nltk()

    st.sidebar.title("Cyberbullying Classifier")
    page = st.sidebar.radio("Navigation", PAGE_OPTIONS)

    model, vectorizer, label_encoder, missing = load_artifacts()
    df = load_dataset()
    evaluation_data = load_evaluation_data()

    if missing:
        st.error("Model artifacts are missing.")
        st.code("python train_and_save_model.py")
        st.write("Missing files:")
        for path in missing:
            st.write(f"- {path}")
        return

    if page == "Tweet Classifier":
        classifier_page(model, vectorizer, label_encoder)
    elif page == "Dataset Overview":
        dataset_overview_page(df)
    elif page == "Model Performance":
        model_performance_page(evaluation_data, label_encoder)
    elif page == "Processing Pipeline":
        processing_pipeline_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
