"""
Doc corpus, tach cau, tokenize va chia train/valid/test.
"""

import json
import random
import re

DATA_PATH = "../lab1/dataset/30k_documents.json"


def load_documents(path=DATA_PATH):
    """Doc document """
    doc = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc.append(json.loads(line)["text"])
    return doc


def doc_to_sen(doc):
    """Tach 1 document thanh list cac cau (string) """
    sentences = []
    for paragraph in doc.splitlines():                         
        for s in re.split(r"(?<=[.!?])\s+", paragraph):        
            if s.strip():                                       
                sentences.append(s.strip())
    return sentences

def tokenize(sentence):
    """Tach 1 cau thanh list token """
    return re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", sentence.lower())


def doc_to_tok(docs):
    """list document -> list cau da tokenize."""
    sents = []
    for text in docs:
        for s in doc_to_sen(text):
            toks = tokenize(s)
            if toks:
                sents.append(toks)
    return sents


def split_train_valid_test(docs, ratios=(0.8, 0.1, 0.1), seed=42):
    """Chia theo docs de cau cung 1 bai khong lot sang tap test."""
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    n = len(docs)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    return docs[:n_train], docs[n_train:n_train + n_valid], docs[n_train + n_valid:]


def prepare_corpus(path=DATA_PATH, seed=42):
    """Tra ve (train_docs, valid_docs, test_docs, train_sents, valid_sents, test_sents)."""
    docs = load_documents(path)
    train_docs, valid_docs, test_docs = split_train_valid_test(docs, seed=seed)
    return (
        train_docs, valid_docs, test_docs,
        doc_to_tok(train_docs),
        doc_to_tok(valid_docs),
        doc_to_tok(test_docs),
    )

