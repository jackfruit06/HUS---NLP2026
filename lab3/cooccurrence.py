import re
import json
from collections import Counter
from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np


DATA_PATH = Path(__file__).parent.parent / "lab1" / "dataset" / "30k_documents.json"


def tokenize(text):
    text = text.lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)

def load_documents(n_docs, path=DATA_PATH):
    docs = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if len(docs) == n_docs:
                break
            docs.append(json.loads(line)['text'])
    return docs


def split_sen(docs):
    sentences = []
    for doc in docs:
        for sen in re.split(r"[.!?\n]+",doc):
            tokens = tokenize(sen)
            if len(tokens) > 1:
                sentences.append(tokens)
    return sentences

def build_vocabulary(sentences, min_count=1):
    count = Counter(w for sen in sentences for w in sen)

    words = sorted([w for w,c in count.items() if c >= min_count])
    vocab = {}
    for i,w in enumerate(words):
        vocab[w] = i
    return vocab

def build_cooccurrence_matrix(sentences, vocab, window=2):
    pairs = Counter()
    for sen in sentences:
        idx = [vocab.get(w) for w in sen]
        for i in range(len(idx)):
            if idx[i] is None:
                continue
            left = max(0,i-window)
            right = min(len(idx), i+window+1)
            for j in range(left,right):
                if j !=i and idx[j] is not None:
                    pairs[(idx[i], idx[j])] += 1

    rows = [r for r,_ in pairs]
    cols = [c for _,c in pairs]
    fre = list(pairs.values())
    n = len(vocab)
    return coo_matrix((fre, (rows, cols)), shape=(n,n), dtype=np.float32).tocsr()

def cosine_similarity(x,y):
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        return 0.0
    return float(np.dot(x,y)/norm)

def most_similar(word, matrix, vocab, top_k=5):
    words = list(vocab)
    idx = vocab[word]
    v = matrix[idx]

    scores = (matrix @ v.T).toarray().ravel()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    similarity = scores / (norms * norms[idx] + 1e-9)

    res = []
    for i in np.argsort(-similarity):
        if i != idx:
            res.append((words[i], round(float(similarity[i]),4)))
        if len(res) == top_k:
            break
    return res

 


