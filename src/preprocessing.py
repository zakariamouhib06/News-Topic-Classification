import pandas as pd
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# =========================
# 1. STOPWORDS MANUELS
# =========================
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}


# =========================
# 2. CLEANING FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)   # ponctuation
    text = re.sub(r"\d+", "", text)        # chiffres
    text = text.strip()
    return text


# =========================
# 3. STOPWORDS + TOKENIZE
# =========================
def preprocess_text(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


# =========================
# 4. LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
print("Data loaded:", df.head())

TEXT_COL = "Text"
LABEL_COL = "Category"


# =========================
# 5. APPLY PREPROCESSING
# =========================
df["clean_text"] = df[TEXT_COL].apply(clean_text)
df["clean_text"] = df["clean_text"].apply(preprocess_text)


# =========================
# 6. FEATURES + LABELS
# =========================
X = df["clean_text"]
y = df[LABEL_COL]


# =========================
# 7. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X_transformed = vectorizer.fit_transform(X)


# =========================
# 8. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y,
    test_size=0.2,
    random_state=42
)

# Crée les dossiers si ils n'existent pas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 9. SAVE VECTORIZER
# =========================
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
print("Vectorizer saved!")


# =========================
# 10. SAVE SPLITS
# =========================
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(DATA_DIR, "splits.pkl"))
print("Preprocessing finished!")