"""
Muc 14-15 - Core implementation: N-gram Language Model
KHONG dung thu vien language model (nltk.lm, kenlm, ...). Chi dung Python thuan + collections/math.

Quy uoc (dung xuyen suot ca lab, de cac model so sanh duoc voi nhau):
    - Moi cau la mot list token, CHUA pad. Vi du: ["the", "cat", "eats", "fish"]
    - Khi dem / tinh xac suat, cau duoc pad: (n-1) x <s> o dau, 1 x </s> o cuoi.
        n=1: ["the", "cat", "</s>"]
        n=2: ["<s>", "the", "cat", "</s>"]
        n=3: ["<s>", "<s>", "the", "cat", "</s>"]
    - <s> chi la context, KHONG bao gio duoc du doan -> khong nam trong vocab.
      </s> va <UNK> duoc du doan -> nam trong vocab.  V = len(vocab).
    - So token N khi tinh perplexity = so tu + 1 (</s>) cho moi cau.

"""

import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>","</s>", "<UNK>"


def build_vocabulary(sentences, min_freq=1):
    """Tra ve set cac tu xuat hien >= min_freq lan trong `sentences`, cong them EOS va UNK """
    counts =  Counter()
    for sen in sentences:
        counts.update(sen)
    vocab = {w for w,c in counts.items() if c >= min_freq}
    vocab.add(UNK)
    vocab.add(EOS)
    return vocab
   

def replace_oov(sentences, vocab):
    """Thay moi token khong nam trong vocab bang UNK """
    return [[w if w in vocab else UNK for w in sen] for sen in sentences]


def pad_sentence(tokens, n):
    """[BOS] * (n-1) + tokens + [EOS]"""
    return [BOS]*(n-1) + list(tokens) + [EOS]


def count_ngrams(sentences, n):
    """Tra ve Counter {tuple n-gram: count} tren cac cau DA PAD """
    counts = Counter()
    for sen in sentences:
        paded = pad_sentence(sen,n)
        for i in range(len(paded) - n + 1):
            counts[tuple(paded[i:i+n])] += 1
    return counts


class NGramLanguageModel:
    def __init__(self, n, smoothing="mle", min_freq=1):
        """
        n         : 1 = unigram, 2 = bigram, 3 = trigram
        smoothing : "mle" hoac "laplace"
        min_freq  : tu xuat hien < min_freq lan trong train se thanh <UNK>
        """
        self.n = n
        self.smoothing = smoothing
        self.min_freq = min_freq
        self.vocab = set()
        self.ngram_counts = Counter()           # {(h..., w): c(h, w)}
        self.context_counts = Counter()         # {(h...): c(h)}   (unigram: key la tuple rong ())
        self.next_word = defaultdict(Counter)   # {(h...): Counter{w: c(h, w)}} -> dung cho next_word_distribution

    def fit(self, corpus):
        """corpus: list cac cau (list token). Tra ve self.

        TODO:
            1. self.vocab = build_vocabulary(corpus, self.min_freq)
            2. corpus = replace_oov(corpus, self.vocab)
            3. self.ngram_counts = count_ngrams(corpus, self.n)
            4. Tu ngram_counts suy ra context_counts va followers:
                   for ng, c in self.ngram_counts.items():
                       h, w = ng[:-1], ng[-1]
                       ...
               (Lam the nay dung cho ca n=1, vi ng[:-1] = () ).
        """
        self.vocab = build_vocabulary(corpus,self.min_freq)
        corpus = replace_oov(corpus,self.vocab)
        self.ngram_counts = count_ngrams(corpus, self.n)

        self.context_counts = Counter()
        self.next_word = defaultdict(Counter)

        for ngram, c in self.ngram_counts.items():
            context_h, predict_w = ngram[:-1], ngram[-1]
            self.context_counts[context_h] += c
            self.next_word[context_h][predict_w] = c
        return self

    def _context_input(self, context):
        """Chuyen context (list token, do dai bat ky) thanh tuple (n-1) token cuoi.

        - Token la BOS giu nguyen; token khong trong vocab -> UNK.
        - Neu context ngan hon n-1 token -> pad them BOS ben trai.
        - n = 1 -> luon tra ve ().
        Vi du trigram: ["x", "the", "cat"] -> ("the", "cat");  ["cat"] -> ("<s>", "cat")
        """
        if self.n == 1:
            return ()
        context = [t if (t == BOS or t in self.vocab) else UNK for t in context]
        context = [BOS] * (self.n - 1) + context
        return tuple(context[-(self.n-1):])

    def probability(self, context, word):
        """P(word | context).
        MLE    : c(h, w) / c(h)               (tra ve 0.0 neu c(h) = 0)
        Laplace: (c(h, w) + 1) / (c(h) + V)
        word khong nam trong vocab -> coi nhu UNK.
        """

        h = self._context_input(context)
        if word not in self.vocab:
            word = UNK
        count_hw = self.ngram_counts.get(h + (word,),0)
        count_h = self.context_counts.get(h,0)

        if self.smoothing == 'mle':
            return count_hw / count_h if count_h else 0.0

        return (count_hw + 1) / (count_h + len(self.vocab))

    def sentence_log_probability(self, sentence):
        """log P(S) = sum_t log P(w_t | context_t), tren cau da pad """
        paded = pad_sentence(sentence, self.n)
        total = 0.0
        for i in range(self.n - 1, len(paded)):
            p = self.probability(paded[i - self.n+1:i], paded[i])
            if p <= 0:
                return -math.inf
            total += math.log(p)
        return total

    def sentence_probability(self, sentence):
        """P(S) = exp(log P(S)). Chi dung de minh hoa - voi cau dai se underflow ve 0.0."""
        return math.exp(self.sentence_log_probability(sentence))


    def perplexity(self, corpus):
        """PP = exp( -(1/N) * sum log P(S) ) """
        total = 0.0
        tokens = 0
        for sen in corpus:
            log_sen = self.sentence_log_probability(sen)
            if log_sen == -math.inf: return math.inf
            total += log_sen
            tokens += len(sen) + 1
        return math.exp(-total / tokens)

    def next_word_distribution(self, context, top_k=None):
        """Tra ve list [(word, prob), ...] sap xep giam dan theo prob; top_k=None -> tra ve het """
        h = self._context_input(context)
        if self.smoothing == 'laplace':
            count_h = self.context_counts.get(h, 0)
            V = len(self.vocab)
            pro_dict = {w: (self.ngram_counts.get(h + (w,), 0) + 1) / (count_h + V) for w in self.vocab}
        else:
            count_h = self.context_counts.get(h,0)
            pro_dict = {w: c_w / count_h for w, c_w in self.next_word.get(h,{}).items()}
        chart = sorted(pro_dict.items(), key=lambda w: w[1], reverse=True)
        return chart[:top_k] if top_k else chart

    def continuation_log_probability(self, context, continuation):
        """log P(continuation | context) """
        hist = list(context)
        total = 0.0
        for w in continuation:
            pro = self.probability(hist, w)
            if pro == 0: return -math.inf
            total += math.log(pro)
            hist.append(w)
        return total


def train_unigram(corpus, **kwargs):
    return NGramLanguageModel(1, **kwargs).fit(corpus)


def train_bigram(corpus, **kwargs):
    return NGramLanguageModel(2, **kwargs).fit(corpus)


def train_trigram(corpus, **kwargs):
    return NGramLanguageModel(3, **kwargs).fit(corpus)
