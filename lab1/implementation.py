"""
Part E - Core Implementation
Tu xay dung mot phien ban TF-IDF toi gian tren mot corpus nho. KHONG su dung truc tiep TfidfVectorizer().

Cac ham chinh:
    build_vocabulary(docs_tokens)
    compute_counts(tokens, vocabulary)
    compute_tf(counts)
    compute_idf(docs_tokens, vocabulary)
    compute_tfidf(tf, idf)
    cosine_similarity(vec1, vec2)

Corpus kiem thu:
    D1 = "cat eats fish"
    D2 = "dog eats fish"
    D3 = "cat likes fish"
"""

import math
from preprocessing import tokenize_minimal

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


def build_vocabulary(docs_tokens):
    """Tra ve danh sach unique terms, sap xep theo alphabet."""
    vocab = set()
    for tokens in docs_tokens:
        vocab.update(tokens)
    return sorted(vocab)


def compute_counts(tokens, vocabulary):
    """Dem so lan xuat hien cua moi term trong vocabulary"""
    counts = {term: 0 for term in vocabulary}
    for tok in tokens:
        if tok in counts:
            counts[tok] += 1
    return counts


def compute_tf(counts):
    """tf(t, d) = c(t, d) / tong so term trong d"""
    total = sum(counts.values())
    if total == 0:
        return {term: 0.0 for term in counts}
    return {term: c / total for term, c in counts.items()}


def compute_idf(docs_tokens, vocabulary):
    """idf(t) = log(N / df(t)) """
    n_docs = len(docs_tokens)
    idf = {}
    for term in vocabulary:
        df = sum(1 for tokens in docs_tokens if term in tokens)
        idf[term] = math.log(n_docs / df) if df > 0 else 0.0
    return idf


def compute_tfidf(tf, idf):
    """tfidf(t, d) = tf(t, d) * idf(t)"""
    return {term: tf[term] * idf.get(term, 0.0) for term in tf}


def cosine_similarity(vec1, vec2):
    """cos(x, y) = (x^T. y) / (||x|| * ||y||)"""
    terms = set(vec1) | set(vec2)
    dot = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in terms)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def run_tests():
    D1 = "cat eats fish"
    D2 = "dog eats fish"
    D3 = "cat likes fish"

    corpus = [D1, D2, D3]

    docs_tokens = [tokenize_minimal(D1), tokenize_minimal(D2), tokenize_minimal(D3)]
    vocabulary = build_vocabulary(docs_tokens)

    # test build_vocabulary
    assert vocabulary == ["cat", "dog", "eats", "fish", "likes"], vocabulary

    # test compute_counts
    counts_d1 = compute_counts(docs_tokens[0], vocabulary)
    assert counts_d1 == {"cat": 1, "dog": 0, "eats": 1, "fish": 1, "likes": 0}

    # test compute_tf: tong tf phai bang 1
    tf_d1 = compute_tf(counts_d1)
    assert abs(sum(tf_d1.values()) - 1.0) < 1e-9
    assert abs(tf_d1["cat"] - 1 / 3) < 1e-9

    # test compute_idf: "fish" xuat hien trong ca 3 doc -> idf = log(3/3) = 0
    idf = compute_idf(docs_tokens, vocabulary)
    assert abs(idf["fish"] - 0.0) < 1e-9
    assert abs(idf["dog"] - math.log(3 / 1)) < 1e-9

    # test compute_tfidf: fish luon co tfidf = 0 vi idf(fish) = 0
    tfidf_d1 = compute_tfidf(tf_d1, idf)
    assert abs(tfidf_d1["fish"] - 0.0) < 1e-9
    assert tfidf_d1["cat"] > 0

    # test cosine_similarity voi vi du trong de bai: x=[1,1,1], y=[1,1,0]
    x = {"a": 1, "b": 1, "c": 1}
    y = {"a": 1, "b": 1, "c": 0}
    expected = 2 / (math.sqrt(3) * math.sqrt(2))
    assert abs(cosine_similarity(x, y) - expected) < 1e-9

    vectorizer = TfidfVectorizer()
    vt = vectorizer.fit_transform(corpus)

    count_vector = CountVectorizer()
    counts = count_vector.fit_transform(corpus).toarray()
    tf = counts/counts.sum(axis=1, keepdims=True)

    print("All tests are PASS\n")

    print('===== Manual =====')
    print("Vocabulary:", vocabulary)
    print("counts(D1):", counts_d1)
    print("TF(D1):", {k: round(v, 4) for k, v in tf_d1.items()})
    print("IDF:", {k: round(v, 4) for k, v in idf.items()})
    print("TF-IDF(D1):", {k: round(v, 4) for k, v in tfidf_d1.items()})

    print('===== Scikit - learn =====')
    print("Vocabulary: ", vectorizer.get_feature_names_out())
    print("counts(D1):", counts[0])
    print("TF (D1): ", tf[0])
    print('IDF: ', dict(zip(vectorizer.get_feature_names_out(),vectorizer.idf_)))
    print("TF-IDF (D1): \n", vt.toarray()[0])


if __name__ == "__main__":
    run_tests()


"""  
===== Manual =====
Vocabulary: ['cat', 'dog', 'eats', 'fish', 'likes']
counts(D1): {'cat': 1, 'dog': 0, 'eats': 1, 'fish': 1, 'likes': 0}
TF(D1): {'cat': 0.3333, 'dog': 0.0, 'eats': 0.3333, 'fish': 0.3333, 'likes': 0.0}
IDF: {'cat': 0.4055, 'dog': 1.0986, 'eats': 0.4055, 'fish': 0.0, 'likes': 1.0986}
TF-IDF(D1): {'cat': 0.1352, 'dog': 0.0, 'eats': 0.1352, 'fish': 0.0, 'likes': 0.0}
===== Scikit - learn =====
Vocabulary:  ['cat' 'dog' 'eats' 'fish' 'likes']
counts(D1): [1 0 1 1 0]
TF (D1):  [0.33333333 0.         0.33333333 0.33333333 0.        ]
IDF:  {'cat': 1.2876820724517808, 'dog': 1.6931471805599454, 'eats': 1.2876820724517808, 'fish': 1.0, 'likes': 1.6931471805599454}
TF-IDF (D1): 
 [0.61980538 0.         0.61980538 0.48133417 0.        ]

So sánh với thư viện:
- Số lượng vocabulary và kết quả tính tuần suất (tf) của của hai các làm đều ra cùng một kết quả.
Thế nhưng sự chênh lệch rõ được thể hiện qua chỉ số IDF và dẫn đến TF-IDF cũng bị lệch theo giữa 2 cách làm
Phân tích nguyên nhân:
- Ở cách làm thủ công chỉ số IDF(fish) = 0 nhưng khi sử dụng thư viện thì chỉ số IDF(fish) = 1 như vậy dựa trên kiến thức được học
có thể thấy công thức ở cả hai cách tính idf đang có sự khác nhau.
Công thức idf thủ công đang dùng:
idf(t) = log(n/df(t))
Công thức idf thư viện scikit-learn dùng:
idf(t) = log((n+1)/(1 + df(t))) + 1
- Thư viện tự động L2-normalize vecto TF-IDF của mỗi document còn cách làm thủ công không có bước chuẩn hoá này

"""