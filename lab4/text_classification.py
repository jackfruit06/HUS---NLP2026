"""
LAB 04 — Text Classification
Core pipeline functions for text classification experiments on IMDb dataset.
"""

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from gensim.models import Word2Vec


DATASET_PATH = "../dataset/IMDB Dataset.csv"


def load_imdb(path=DATASET_PATH):
    """Load IMDB Dataset from CSV."""
    df = pd.read_csv(path)
    return df


def inspect_dataset(df):
    """
    Experiment 1 — Dataset inspection.
    Returns a dict with basic statistics.
    """
    info = {}
    info["n_samples"] = len(df)
    info["n_classes"] = df["sentiment"].nunique()
    info["classes"] = df["sentiment"].unique().tolist()

    # class distribution
    class_dist = df["sentiment"].value_counts()
    info["class_distribution"] = class_dist.to_dict()
    info["class_percentage"] = (class_dist / len(df) * 100).round(2).to_dict()

    # document length (word count)
    lengths = df["review"].apply(lambda x: len(str(x).split()))
    info["length_stats"] = lengths.describe().to_dict()

    # duplicates
    info["n_duplicates"] = df["review"].duplicated().sum()

    # missing values
    info["missing_text"] = df["review"].isna().sum()
    info["missing_labels"] = df["sentiment"].isna().sum()

    return info


def print_inspection(info):
    """Pretty-print dataset inspection results."""
    print(f"Số samples: {info['n_samples']}")
    print(f"Số classes: {info['n_classes']}")
    print(f"Classes: {info['classes']}")
    print()

    print("Class distribution:")
    table = pd.DataFrame({
        "Class": list(info["class_distribution"].keys()),
        "Count": list(info["class_distribution"].values()),
        "Percentage": [f"{v}%" for v in info["class_percentage"].values()],
    })
    print(table.to_string(index=False))
    print()

    print(f"Duplicate documents: {info['n_duplicates']}")
    print(f"Missing text: {info['missing_text']}")
    print(f"Missing labels: {info['missing_labels']}")
    print()

    print("Document length statistics:")
    for k, v in info["length_stats"].items():
        print(f"  {k}: {v:.2f}")


def plot_class_distribution(df):
    """Plot class distribution bar chart."""
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Frequency")
    ax.set_title("Class Distribution — IMDb Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_length_distribution(df):
    """Plot document length histogram."""
    lengths = df["review"].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7, color="#3498db")
    ax.set_xlabel("Document length (number of words)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Distribution of Document Length")
    plt.tight_layout()
    plt.show()


def encode_labels(df):
    """Encode sentiment labels: positive -> 1, negative -> 0."""
    label_map = {"positive": 1, "negative": 0}
    return df["sentiment"].map(label_map).values


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train / validation / test.
    """
    X = df["review"].values
    y = encode_labels(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=random_state, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tfidf(X_train, X_val, X_test, ngram_range=(1, 1), max_features=None):
    """
    Fit TF-IDF on training set, transform all splits.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)  # fit on train only
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF ngram_range={ngram_range}, max_features={max_features}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train matrix shape: {X_train_tfidf.shape}")

    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf


def tokenize(text):
    """Simple tokenization: lowercase + word extraction."""
    text = str(text).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def train_word2vec(X_train, vector_size=100, window=5, min_count=2, epochs=10):
    """Train Word2Vec on training corpus."""
    sentences = [tokenize(doc) for doc in X_train]
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
    )
    print(f"Word2Vec trained: vocab={len(model.wv)}, dim={vector_size}")
    return model


def doc_to_avg_vec(doc, w2v_model, vector_size):
    """Convert a document to its average Word2Vec vector."""
    tokens = tokenize(doc)
    vectors = []
    for token in tokens:
        if token in w2v_model.wv:
            vectors.append(w2v_model.wv[token])
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def build_avg_word2vec(X_train, X_val, X_test, w2v_model, vector_size=100):
    """Build average Word2Vec features for all splits."""
    X_train_w2v = np.array([doc_to_avg_vec(d, w2v_model, vector_size) for d in X_train])
    X_val_w2v = np.array([doc_to_avg_vec(d, w2v_model, vector_size) for d in X_val])
    X_test_w2v = np.array([doc_to_avg_vec(d, w2v_model, vector_size) for d in X_test])

    print(f"Average Word2Vec: Train shape={X_train_w2v.shape}")
    return X_train_w2v, X_val_w2v, X_test_w2v



def train_logistic_regression(X_train, y_train, max_iter=1000):
    """Train Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Logistic Regression trained in {elapsed:.2f}s")
    return clf, elapsed


def train_linear_svm(X_train, y_train, max_iter=1000):
    """Train Linear SVM classifier."""
    clf = LinearSVC(max_iter=max_iter, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Linear SVM trained in {elapsed:.2f}s")
    return clf, elapsed


def evaluate(clf, X_test, y_test, model_name="Model"):
    """Evaluate a classifier and return metrics dict."""
    y_pred = clf.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="binary"),
        "Recall": recall_score(y_test, y_pred, average="binary"),
        "F1": f1_score(y_test, y_pred, average="binary"),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")
    print()

    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = ["Negative", "Positive"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Analysis
    TP, FN = cm[1][1], cm[1][0]
    FP, TN = cm[0][1], cm[0][0]
    print(f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    if FP > FN:
        print("→ False Positives nhiều hơn: model có xu hướng predict Positive quá nhiều.")
    elif FN > FP:
        print("→ False Negatives nhiều hơn: model bỏ sót nhiều Positive thực tế.")
    else:
        print("→ FP và FN cân bằng.")

    return cm


def error_analysis(X_test, y_test, y_pred, n_correct=3, n_incorrect=3):
    label_names = {0: "negative", 1: "positive"}

    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask

    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(incorrect_mask)[0]

    rng = np.random.RandomState(42)
    chosen_correct = rng.choice(correct_idx, size=min(n_correct, len(correct_idx)), replace=False)
    chosen_incorrect = rng.choice(incorrect_idx, size=min(n_incorrect, len(incorrect_idx)), replace=False)

    rows = []
    for idx in chosen_correct:
        text = X_test[idx]
        short_text = text[:200] + "..." if len(text) > 200 else text
        rows.append({
            "Text": short_text,
            "True": label_names[y_test[idx]],
            "Predicted": label_names[y_pred[idx]],
            "Correct": True,
        })

    for idx in chosen_incorrect:
        text = X_test[idx]
        short_text = text[:200] + "..." if len(text) > 200 else text
        rows.append({
            "Text": short_text,
            "True": label_names[y_test[idx]],
            "Predicted": label_names[y_pred[idx]],
            "Correct": False,
        })

    return pd.DataFrame(rows)


def word_order_experiment(w2v_model, vector_size=100):
    s1 = "this movie is good"
    s2 = "this movie is not good"

    v1 = doc_to_avg_vec(s1, w2v_model, vector_size)
    v2 = doc_to_avg_vec(s2, w2v_model, vector_size)

    # cosine similarity
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

    print(f'S1 = "{s1}"')
    print(f'S2 = "{s2}"')
    print(f"Cosine similarity of average embeddings: {cos_sim:.4f}")
    print()

    if cos_sim > 0.9:
        print("→ Cosine similarity cao!")
        print("  Average word embeddings KHÔNG thể phân biệt được vai trò của 'not'.")
        print("  Đây là limitation cơ bản của bag-of-words và averaged embeddings:")
        print("  chúng mất hoàn toàn word order và negation information.")
    else:
        print("→ Cosine similarity thấp hơn mong đợi, nhưng representation")
        print("  vẫn không thực sự capture được negation một cách chính xác.")

    return cos_sim


def inspect_model_weights(clf, vectorizer, top_n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = clf.coef_[0]

    # Top positive features (w_i >> 0)
    top_pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_pos = pd.DataFrame({
        "Feature": feature_names[top_pos_idx],
        "Weight": coefficients[top_pos_idx].round(4),
    })

    # Top negative features (w_i << 0)
    top_neg_idx = np.argsort(coefficients)[:top_n]
    top_neg = pd.DataFrame({
        "Feature": feature_names[top_neg_idx],
        "Weight": coefficients[top_neg_idx].round(4),
    })

    print("Top POSITIVE features (contribute to positive class):")
    print(top_pos.to_string(index=False))
    print()
    print("Top NEGATIVE features (contribute to negative class):")
    print(top_neg.to_string(index=False))
    print()

    return top_pos, top_neg


def plot_top_features(clf, vectorizer, top_n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = clf.coef_[0]

    top_pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefficients)[:top_n]

    idx = np.concatenate([top_neg_idx, top_pos_idx])
    colors = ["#e74c3c"] * top_n + ["#2ecc71"] * top_n

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), coefficients[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(feature_names[idx])
    ax.set_xlabel("Weight")
    ax.set_title("Top Positive & Negative Features (Logistic Regression + TF-IDF)")
    plt.tight_layout()
    plt.show()


def comparison_table(results_list):
    """Build a comparison DataFrame from a list of metric dicts."""
    df = pd.DataFrame(results_list)
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df
