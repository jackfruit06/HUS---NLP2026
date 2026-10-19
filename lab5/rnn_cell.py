"""
Muc 14 - Core implementation: Simple RNN Cell & Sequence Pipeline

Tu implement mot RNN cell bang NumPy de hieu co che forward computation:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

Muc tieu: Hieu computation graph cua recurrent model, KHONG phai thay the PyTorch.

Quy uoc (dung xuyen suot ca lab, de cac model so sanh duoc voi nhau):
    - Input x_t la vector embedding co chieu input_size.
    - Hidden state h_t la vector co chieu hidden_size, khoi tao zeros.
    - Forward_sequence cap nhat hidden state lien tuc qua T timestep, tra ve H = [h_1, ..., h_T].
    - Weight khoi tao Xavier uniform de on dinh gradient.
    - Tokenizer: lowercase, loai bo html <br />, tach tu regex [a-z0-9]+(?:'[a-z]+)?.
    - Vocabulary: fit tren TRAIN only; token 0 = <PAD>, token 1 = <UNK>.
    - Split: 70% train, 15% validation, 15% test, stratify theo label, seed=42.

Cac class va ham chinh:
    SimpleRNNCell(input_size, hidden_size, seed=42)
    verify_rnn_cell()
    calculation_exercise()
    clean_text(text)
    tokenize(text)
    load_imdb(path, drop_duplicates=True)
    split_data(texts, labels, seed=42, ratios=(0.7, 0.15, 0.15))
    Vocabulary(min_freq=2, max_size=None)
    IMDbSequenceDataset(texts, labels, vocab, max_length=200)
    compute_token_retention(texts, vocab, max_length)
    compute_sequence_lengths(texts, vocab)
    SequenceClassifier(vocab_size, embed_dim, hidden_size, num_classes=2, rnn_type='lstm')
    train_model(model, train_loader, valid_loader, n_epochs=10, lr=1e-3)
    get_predictions(model, loader)
    predict_review(text, model, vocab, max_length=200)
    evaluate(y_true, y_pred)
    confusion_pos_first(y_true, y_pred)
    metrics_from_confusion(cm)
    save_result(exp, setting, metric, value, filename="results.csv")
"""

import collections
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = "../dataset/IMDB Dataset.csv"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ===========================================================================
# 1. Core Implementation: SimpleRNNCell (NumPy thuần)
# ===========================================================================

class SimpleRNNCell:
    """
    Mot RNN cell don gian, implement hoan toan bang NumPy.

    Computation:
        a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h
        h_t = tanh(a_t)
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        # Xavier / Glorot uniform initialization
        limit_xh = np.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))

        self.W_xh = rng.uniform(-limit_xh, limit_xh, size=(hidden_size, input_size)).astype(np.float32)
        self.W_hh = rng.uniform(-limit_hh, limit_hh, size=(hidden_size, hidden_size)).astype(np.float32)
        self.b_h = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x_t, h_prev):
        """
        Mot buoc forward tai timestep t:
            a_t = W_xh @ x_t + W_hh @ h_prev + b_h
            h_t = tanh(a_t)
        """
        x_t = np.asarray(x_t, dtype=np.float32)
        h_prev = np.asarray(h_prev, dtype=np.float32)

        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(a_t)
        return h_t

    def forward_sequence(self, X, h_0=None):
        """
        Chay RNN cell qua toan bo sequence X co shape [T, input_size].
        Tra ve H co shape [T, hidden_size].
        """
        T = X.shape[0]
        if h_0 is None:
            h_0 = np.zeros(self.hidden_size, dtype=np.float32)

        H = np.zeros((T, self.hidden_size), dtype=np.float32)
        h = h_0.copy()
        for t in range(T):
            h = self.forward(X[t], h)
            H[t] = h
        return H

    def initial_hidden(self):
        """Tra ve vector hidden state khoi dau (zeros)."""
        return np.zeros(self.hidden_size, dtype=np.float32)

    def __repr__(self):
        return (
            f"SimpleRNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})\n"
            f"  W_xh: {self.W_xh.shape}  W_hh: {self.W_hh.shape}  b_h: {self.b_h.shape}"
        )


def verify_rnn_cell(input_size=8, hidden_size=5, T=6, seed=7):
    """
    Doi chieu SimpleRNNCell (NumPy) voi PyTorch nn.RNNCell tren cung mot sequence.
    Tra ve (max_diff, H_np, H_torch).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch chua duoc cai dat. Bo qua phan doi chieu framework.")
        return 0.0, None, None

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, input_size)).astype(np.float32)
    h_0 = rng.standard_normal(hidden_size).astype(np.float32)

    # 1. NumPy
    cell_np = SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, seed=seed)
    H_np = cell_np.forward_sequence(X, h_0=h_0)

    # 2. PyTorch
    torch_cell = nn.RNNCell(input_size, hidden_size, bias=True)
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.from_numpy(cell_np.W_xh))
        torch_cell.weight_hh.copy_(torch.from_numpy(cell_np.W_hh))
        torch_cell.bias_ih.copy_(torch.from_numpy(cell_np.b_h))
        torch_cell.bias_hh.zero_()

    torch_cell.eval()
    H_torch = np.zeros((T, hidden_size), dtype=np.float32)
    h_t = torch.from_numpy(h_0)

    with torch.no_grad():
        for t in range(T):
            x_t = torch.from_numpy(X[t])
            h_t = torch_cell(x_t, h_t)
            H_torch[t] = h_t.numpy()

    max_diff = float(np.max(np.abs(H_np - H_torch)))
    return max_diff, H_np, H_torch


def calculation_exercise():
    """
    Tinh toan so hoc cho Bai 1 (tinh tay mot buoc forward RNN theo dung de bai muc 6).
    x_t = [1.0, 0.0]^T, h_{t-1} = [0.5, 0.2]^T
    W_xh = [[0.5, 0.2], [0.1, 0.4]]
    W_hh = [[0.3, 0.1], [0.2, 0.5]]
    b_h  = [0.0, 0.0]^T
    """
    x_t = np.array([1.0, 0.0], dtype=np.float32)
    h_prev = np.array([0.5, 0.2], dtype=np.float32)

    W_xh = np.array([[0.5, 0.2],
                     [0.1, 0.4]], dtype=np.float32)
    W_hh = np.array([[0.3, 0.1],
                     [0.2, 0.5]], dtype=np.float32)
    b_h = np.array([0.0, 0.0], dtype=np.float32)

    a_t = W_xh @ x_t + W_hh @ h_prev + b_h
    h_t = np.tanh(a_t)

    print("=== Tinh toan forward RNN (Bai 1 theo de bai muc 6) ===")
    print(f"W_xh @ x_t     = {W_xh @ x_t}")
    print(f"W_hh @ h_{{t-1}} = {W_hh @ h_prev}")
    print(f"b_h            = {b_h}")
    print(f"a_t            = {a_t}  (khop [0.67, 0.30])")
    print(f"h_t = tanh(a_t)= {h_t}  (khop [0.5848, 0.2913])")

    return {"a_t": a_t, "h_t": h_t}


# ===========================================================================
# 2. Preprocessing & Data Loading
# ===========================================================================

def clean_text(text):
    """Bo the HTML <br /> co trong review IMDb va chuan hoa khoang trang."""
    text = str(text).replace("<br />", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Tach tu: lowercase, giu lai a-z, chu so va apostrophe."""
    text = clean_text(text).lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def load_imdb(path=DATA_PATH, drop_duplicates=True):
    """Tra ve DataFrame [text, label]; label: 1 = positive, 0 = negative."""
    df = pd.read_csv(path)
    df = pd.DataFrame({
        "text": df["review"].map(clean_text),
        "label": (df["sentiment"] == "positive").astype(int),
    })
    if drop_duplicates:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def split_data(texts, labels, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Chia train / valid / test (stratify theo label)."""
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)

    X_train, X_rest, y_train, y_rest = train_test_split(
        texts, labels, train_size=ratios[0], random_state=seed, stratify=labels
    )
    split_ratio = ratios[1] / (ratios[1] + ratios[2])
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rest, y_rest, train_size=split_ratio, random_state=seed, stratify=y_rest
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


class Vocabulary:
    """
    Vocabulary fit tren training corpus.
    Token 0 = <PAD>, Token 1 = <UNK>.
    """

    def __init__(self, min_freq=2, max_size=None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {}
        self.idx2token = []
        self._built = False

    def build(self, texts):
        counter = collections.Counter()
        for text in texts:
            counter.update(tokenize(text))

        filtered = [w for w, c in counter.items() if c >= self.min_freq]
        filtered.sort(key=lambda w: (-counter[w], w))

        if self.max_size is not None:
            filtered = filtered[: self.max_size - 2]

        self.idx2token = [PAD_TOKEN, UNK_TOKEN] + filtered
        self.token2idx = {tok: idx for idx, tok in enumerate(self.idx2token)}
        self._built = True

    @property
    def pad_idx(self):
        return 0

    @property
    def unk_idx(self):
        return 1

    def encode(self, text):
        """Chuyen text thanh list token IDs."""
        if not self._built:
            raise RuntimeError("Vocabulary chua duoc build! Goi .build(texts) truoc.")
        tokens = tokenize(text)
        unk = self.unk_idx
        return [self.token2idx.get(tok, unk) for tok in tokens]

    def decode(self, ids):
        """Chuyen list token IDs thanh text."""
        return " ".join(self.idx2token[i] if 0 <= i < len(self.idx2token) else UNK_TOKEN for i in ids)

    def __len__(self):
        return len(self.idx2token)


def compute_sequence_lengths(texts, vocab):
    """Do sequence length (so token) cho moi document."""
    return np.array([len(vocab.encode(t)) for t in texts], dtype=np.int32)


def compute_token_retention(texts, vocab, max_length):
    """Tinh ti le phan tram token duoc giu lai khi truncate ve max_length."""
    total_tokens = 0
    retained = 0
    for text in texts:
        n_tok = len(vocab.encode(text))
        total_tokens += n_tok
        retained += min(n_tok, max_length)
    return retained / total_tokens if total_tokens > 0 else 0.0


# ===========================================================================
# 3. Dataset & PyTorch Model Wrappers
# ===========================================================================

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class IMDbSequenceDataset(Dataset):
        def __init__(self, texts, labels, vocab, max_length=200, truncate_side="right"):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_length = max_length
            self.truncate_side = truncate_side

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            ids = self.vocab.encode(self.texts[idx])
            if len(ids) > self.max_length:
                ids = ids[:self.max_length] if self.truncate_side == "right" else ids[-self.max_length:]
            ids = ids + [self.vocab.pad_idx] * (self.max_length - len(ids))
            return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    class SequenceClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, num_classes=2, rnn_type="lstm", num_layers=1, dropout=0.3, pad_idx=0):
            super().__init__()
            self.rnn_type = rnn_type.lower()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.embed_dropout = nn.Dropout(dropout)
            rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
            self.rnn = rnn_cls(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.classifier_dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        
        def forward(self, x):
            # Dem so token thuc cua moi sequence, khong tinh token PAD.
            # Dataset dang padding o cuoi sequence (right padding).
            lengths = x.ne(self.embedding.padding_idx).sum(dim=1)
            lengths = lengths.clamp(min=1).cpu()

            # Chuyen token IDs thanh embedding.
            emb = self.embed_dropout(self.embedding(x))

            # Bo qua cac timestep padding khi chay recurrent layer.
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                emb,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            # Lay hidden state tai token thuc cuoi cung.
            if self.rnn_type == "lstm":
                _, (h_n, _) = self.rnn(packed_emb)
            else:
                _, h_n = self.rnn(packed_emb)

            last_hidden = h_n[-1]

            # Phan loai sentiment.
            logits = self.fc(self.classifier_dropout(last_hidden))
            return logits


        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train_epoch(model, loader, optimizer, criterion, device=DEVICE):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X_b.size(0)
        return total_loss / len(loader.dataset)

    def evaluate_epoch(model, loader, criterion, device=DEVICE):
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                total_loss += criterion(logits, y_b).item() * X_b.size(0)
                correct += (logits.argmax(dim=1) == y_b).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    def train_model(model, train_loader, valid_loader, n_epochs=10, lr=1e-3, device=DEVICE, verbose=True):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_losses": [], "valid_losses": [], "valid_accs": [], "training_time": 0.0}
        t_start = time.time()
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr_l = train_epoch(model, train_loader, optimizer, criterion, device)
            va_l, va_a = evaluate_epoch(model, valid_loader, criterion, device)
            history["train_losses"].append(tr_l)
            history["valid_losses"].append(va_l)
            history["valid_accs"].append(va_a)
            if verbose:
                print(f"Epoch {epoch:>2}/{n_epochs} | Train Loss: {tr_l:.4f} | Val Loss: {va_l:.4f} | Val Acc: {va_a*100:.2f}% | Time: {time.time()-t0:.1f}s")
        history["training_time"] = time.time() - t_start
        return history

    def get_predictions(model, loader, device=DEVICE):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                logits = model(X_b.to(device))
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_b.numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def predict_review(text, model, vocab, max_length=200, device=DEVICE):
        model.eval()
        ids = vocab.encode(text)
        if len(ids) > max_length:
            ids = ids[:max_length]
        ids = ids + [vocab.pad_idx] * (max_length - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0, 1].item()
        return ("positive" if prob >= 0.5 else "negative"), prob

except ImportError:
    DEVICE = "cpu"
    IMDbSequenceDataset = None
    SequenceClassifier = None


# ===========================================================================
# 4. Evaluation & Results Storage
# ===========================================================================

def evaluate(y_true, y_pred):
    """Accuracy, Precision, Recall, F1 (lop positive)."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_pos_first(y_true, y_pred):
    """Confusion matrix theo layout de: hang = Actual (Pos, Neg), cot = Predicted (Pos, Neg)."""
    return confusion_matrix(y_true, y_pred, labels=[1, 0])


def metrics_from_confusion(cm):
    """Tinh lai 4 metric tu [[TP, FN], [FP, TN]] bang cong thuc trong de."""
    (tp, fn), (fp, tn) = cm
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def save_result(exp, setting, metric, value, filename="results.csv"):
    """Luu ket qua theo format chuan (experiment, setting, metric, value) giong cac lab truoc."""
    row = pd.DataFrame([{"experiment": exp, "setting": setting, "metric": metric, "value": value}])
    try:
        old = pd.read_csv(filename)
        updated = pd.concat([old, row], ignore_index=True)
    except FileNotFoundError:
        updated = row
    updated.to_csv(filename, index=False)


if __name__ == "__main__":
    print("Kiem tra SimpleRNNCell:")
    cell = SimpleRNNCell(input_size=4, hidden_size=3)
    print(cell)

    max_diff, _, _ = verify_rnn_cell()
    print(f"\nMax diff voi PyTorch: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Verification failed: max_diff={max_diff}"
    print("SimpleRNNCell verification: PASSED!\n")

    calculation_exercise()
