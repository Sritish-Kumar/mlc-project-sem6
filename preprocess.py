import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


_NLTK_CHECKED = False
_STOP_WORDS = None
_LEMMATIZER = None
_WORDNET_AVAILABLE = False


def ensure_nltk_data() -> None:
    """Download required NLTK resources once when they are missing."""
    global _NLTK_CHECKED, _STOP_WORDS, _LEMMATIZER, _WORDNET_AVAILABLE

    if _NLTK_CHECKED:
        return

    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass

    try:
        _STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        _STOP_WORDS = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "you",
            "your",
        }

    _LEMMATIZER = WordNetLemmatizer()
    try:
        _LEMMATIZER.lemmatize("tests")
        _WORDNET_AVAILABLE = True
    except LookupError:
        _WORDNET_AVAILABLE = False

    _NLTK_CHECKED = True


def clean_text(text: str) -> str:
    ensure_nltk_data()

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    words = []
    for word in text.split():
        if word in _STOP_WORDS:
            continue
        if _WORDNET_AVAILABLE:
            word = _LEMMATIZER.lemmatize(word)
        words.append(word)

    return " ".join(words)


DISPLAY_NAMES = {
    "age": "Age",
    "ethnicity": "Ethnicity",
    "gender": "Gender",
    "religion": "Religion",
    "other_cyberbullying": "Other Cyberbullying",
    "not_cyberbullying": "Not Cyberbullying",
}

CLASS_ORDER = [
    "age",
    "ethnicity",
    "gender",
    "religion",
    "other_cyberbullying",
    "not_cyberbullying",
]
