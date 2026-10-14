"""
Core implementation: Text Classification pipeline (IMDb sentiment)
Pipeline dung (KHONG leakage):
    Dataset -> Train / Validation / Test split -> fit preprocessing (TF-IDF, Word2Vec) tren TRAIN
            -> transform Validation / Test -> train model -> evaluation

Quy uoc (dung xuyen suot ca lab, de cac model so sanh duoc voi nhau):
    - Label: 1 = positive, 0 = negative. Lop duong (positive) la lop dung de tinh precision / recall / F1.
    - Cung mot ham tokenize cho TF-IDF va Word2Vec -> khac biet ket qua chi den tu representation.
    - Moi split chia 80/10/10, stratify theo label, co seed.
"""

import re
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA_PATH = "../dataset/IMDB Dataset.csv"


def clean_text(text):
    """Bo the HTML <br /> co trong review IMDb"""
    return re.sub(r"\s+", " ", text.replace("<br />", " ")).strip()


def tokenize(text):
    text = text.lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def tokenize_no_stopwords(text):
    """Loại bo stopwords (danh sach English cua scikit-learn)"""
    return [w for w in tokenize(text) if w not in ENGLISH_STOP_WORDS]


def load_imdb(path=DATA_PATH, drop_duplicates=True):
    """Tra ve DataFrame [text, label]; label: 1 = positive, 0 = negative"""
    df = pd.read_csv(path)
    df = pd.DataFrame({"text": df["review"].map(clean_text), "label": (df["sentiment"] == "positive").astype(int)})
    if drop_duplicates:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def split_data(texts, labels, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Chia train / valid / test (stratify theo label)"""
    texts, labels = np.asarray(texts, dtype=object), np.asarray(labels)
    X_train, X_rest, y_train, y_rest = train_test_split(
        texts, labels, train_size=ratios[0], random_state=seed, stratify=labels)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rest, y_rest, train_size=ratios[1] / (ratios[1] + ratios[2]), random_state=seed, stratify=y_rest)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def make_tfidf(ngram_range=(1, 1), min_df=2, max_features=None, remove_stopwords=False):
    """TfidfVectorizer dung chung tokenizer cua lab phai fit tren TRAIN."""
    return TfidfVectorizer(tokenizer=tokenize_no_stopwords if remove_stopwords else tokenize,
                           lowercase=False, token_pattern=None, ngram_range=ngram_range,
                           min_df=min_df, max_features=max_features)


def train_word2vec(token_lists, vector_size=100, window=5, min_count=5, sg=1, epochs=5, seed=42, workers=1):
    """Train Word2Vec tren cac document cua TRAIN. workers=1 de ket qua tai lap duoc (cung seed -> cung vector)."""
    return Word2Vec(sentences=token_lists, vector_size=vector_size, window=window, min_count=min_count,
                    sg=sg, negative=5, epochs=epochs, workers=workers, seed=seed)


def doc_vector(tokens, wv):
    """Average Word2Vec: x = (1/n) * sum_i w_i (bo qua tu ngoai vocab; document khong co tu nao -> vector 0)."""
    vecs = [wv[w] for w in tokens if w in wv]
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def docs_to_matrix(token_lists, wv):
    return np.array([doc_vector(t, wv) for t in token_lists])


def make_classifier(kind, C=1.0):
    """kind: 'lr' (Logistic Regression) hoac 'svm' (Linear SVM)."""
    if kind == "lr":
        return LogisticRegression(C=C, max_iter=2000)
    if kind == "svm":
        return LinearSVC(C=C)
    raise ValueError(kind)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_with_valid(X_train, y_train, X_valid, y_valid, kind="lr", grid=(0.1, 1, 10, 100), scale=False):
    """Chon C tot nhat theo F1 tren VALIDATION, tra ve (model, scaler, best_C, train_time).
    scale=True: StandardScaler fit tren train (dung cho dense Word2Vec de LR hoi tu)."""
    scaler = StandardScaler().fit(X_train) if scale else None
    prep = (lambda X: scaler.transform(X)) if scale else (lambda X: X)
    best = (-1, None)
    for C in grid:
        clf = make_classifier(kind, C).fit(prep(X_train), y_train)
        score = f1_score(y_valid, clf.predict(prep(X_valid)))
        if score > best[0]:
            best = (score, C)
    start = time.time()
    clf = make_classifier(kind, best[1]).fit(prep(X_train), y_train)
    return clf, scaler, best[1], time.time() - start


def evaluate(y_true, y_pred):
    """Accuracy, Precision, Recall, F1 (lop positive)."""
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)}


def confusion_pos_first(y_true, y_pred):
    """Confusion matrix theo layout cua de: hang = Actual (Pos, Neg), cot = Predicted (Pos, Neg) -> [[TP, FN], [FP, TN]]."""
    return confusion_matrix(y_true, y_pred, labels=[1, 0])


def metrics_from_confusion(cm):
    """Tinh lai 4 metric tu [[TP, FN], [FP, TN]] bang cong thuc trong de (de doi chieu voi sklearn)."""
    (tp, fn), (fp, tn) = cm
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def explain_prediction(text, vectorizer, clf, k=3):
    """Muc 23 - Application: phan loai 1 review moi.
    Tra ve (label, P(positive), top-k feature keo ve positive, top-k feature keo ve negative) theo x_i*w_i."""
    x = vectorizer.transform([clean_text(text)])
    p = float(clf.predict_proba(x)[0, 1])
    names = vectorizer.get_feature_names_out()
    contrib = x.data * clf.coef_.ravel()[x.indices]
    order = np.argsort(contrib)
    pos = [names[x.indices[j]] for j in order[::-1][:k] if contrib[j] > 0]
    neg = [names[x.indices[j]] for j in order[:k] if contrib[j] < 0]
    return ("positive" if p >= 0.5 else "negative"), p, pos, neg


def top_features(vectorizer, clf, k=10):
    """Top-k feature co weight duong nhat va am nhat cua model tuyen tinh tren TF-IDF."""
    names = np.array(vectorizer.get_feature_names_out())
    w = clf.coef_.ravel()
    order = np.argsort(w)
    pos = [(names[i], float(w[i])) for i in order[::-1][:k]]
    neg = [(names[i], float(w[i])) for i in order[:k]]
    return pos, neg
