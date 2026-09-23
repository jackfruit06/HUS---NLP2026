import re
from nltk.corpus import stopwords

stopword = stopwords.words('english')

suffix = [
    "er", "or", "ist", "ee", "ian", "ant", "ent", "al","tion", "sion", "ment", "ness", "ity", "ty", "ship", "hood", "ance", "ence", "al", "ism", "th",
    "ful", "less", "able", "ible", "ous", "ious", "ive", "ic", "ish", "ary", "ory", "en", "ed", "ing","ize", "ise", "ify", "ate","ly"
]
def punctuation_normalization(text):
    """Thay moi ky tu khong phai chu/so bang khoang trang."""
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def subword(word):
    """Tach 1 tu thanh 2 'subword' bang cach tac hau to."""
    if len(word) <= 4:
        return [word]
    for suf in suffix:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            origin = word[: -len(suf)]
            return [origin, "-" + suf]
    return [word]

def tokenize_minimal(text):
    """Pipeline A: raw text -> lowercase -> tokenization"""
    text = text.lower()
    return text.split()


def tokenize_normalized(text):
    """Pipeline B: raw text -> lowercase -> punctuation normalization(chuẩn hoá dấu câu) -> tokenization -> stopword handling"""
    text = text.lower()
    text = punctuation_normalization(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopword]
    return tokens


def tokenize_extended(text):
    """Pipeline C: raw text -> normalization -> subword tokenization"""
    text = text.lower()
    text = punctuation_normalization(text)
    words = text.split()
    token_subword = []
    for w in words:
        token_subword.extend(subword(w))
    return token_subword
