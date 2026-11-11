"""LAB 08: BERT-Tiny từ đầu, frozen encoder và fine-tuning trên IMDb."""

import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import copy
import gc
import hashlib
import io
import json
import platform
import random
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import BertConfig, BertModel, BertTokenizerFast

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parent
CACHE = ROOT / '.lab08_cache'
MODEL_ID = 'prajjwal1/bert-tiny'
REVISION = '6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
MODEL_DIR = CACHE / 'bert-tiny'
DOMAIN_URL = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
SEED, BASE_LENGTH, BATCH_SIZE, EPOCHS, LR = 42, 128, 64, 3, 2e-4
DEVICE = torch.device('cpu')
INPUT_KEYS = ('input_ids', 'attention_mask', 'token_type_ids')


def configure():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def ensure_assets():
    """Tải các file công khai khi cache chưa có; cố định revision model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'vocab.txt', 'pytorch_model.bin'):
        path = MODEL_DIR / name
        if not path.exists():
            response = requests.get(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}', timeout=180)
            response.raise_for_status()
            path.write_bytes(response.content)
    domain_path = CACHE / 'amazon_cells_labelled.txt'
    if not domain_path.exists():
        response = requests.get(DOMAIN_URL, timeout=120)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            domain_path.write_bytes(archive.read('sentiment labelled sentences/amazon_cells_labelled.txt'))
    return MODEL_DIR


def load_tokenizer():
    ensure_assets()
    return BertTokenizerFast.from_pretrained(MODEL_DIR, do_lower_case=True, local_files_only=True)


def clean_text(text):
    return re.sub(r'\s+', ' ', str(text).replace('<br />', ' ')).strip()


def fingerprint(values):
    payload = json.dumps(list(values), ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def prepare_data():
    path = ROOT / 'dataset' / 'IMDB Dataset.csv'
    raw = pd.read_csv(path)
    if raw[['review', 'sentiment']].isna().any().any():
        raise ValueError('IMDb chứa dữ liệu thiếu.')
    frame = pd.DataFrame({'text': raw.review.map(clean_text), 'label': (raw.sentiment == 'positive').astype(int),
                          'source_row': raw.index}).drop_duplicates('text').reset_index(drop=True)
    train_full, rest = train_test_split(np.arange(len(frame)), train_size=.70, random_state=SEED, stratify=frame.label)
    valid_full, test_full = train_test_split(rest, train_size=.5, random_state=SEED, stratify=frame.label.iloc[rest])
    subset = np.random.default_rng(SEED).choice(len(train_full), 10000, replace=False)
    split_indices = {'train': train_full[subset], 'validation': valid_full[:2000], 'test': test_full[:2000]}
    data = {name: frame.iloc[idx].reset_index(drop=True) for name, idx in split_indices.items()}
    source_sets = [set(data[name].source_row) for name in ('train', 'validation', 'test')]
    assert not (source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2] or source_sets[1] & source_sets[2])
    info = {'dataset': 'IMDb Dataset.csv', 'raw_samples': len(raw), 'clean_samples': len(frame),
            'removed_duplicates': len(raw)-len(frame), 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'split_hashes': {name: fingerprint(df.source_row.tolist()) for name, df in data.items()}}
    return data, info


class PretrainedSentimentClassifier(nn.Module):
    """BERT encoder + [CLS] + classification head cho sentiment.

    frozen/finetuned dùng encoder pretrained; scratch khởi tạo encoder ngẫu nhiên.
    forward trả class logits; predict_text áp dụng softmax và chọn nhãn.
    """
    def __init__(self, mode='finetuned'):
        super().__init__()
        if mode not in ('scratch', 'frozen', 'finetuned'):
            raise ValueError('mode: scratch, frozen hoặc finetuned')
        self.mode = mode
        config = BertConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        config._attn_implementation = 'eager'
        # Transformer encoder: Input IDs -> Contextual representations.
        if mode == 'scratch':
            self.encoder = BertModel(config, add_pooling_layer=False)
        else:
            self.encoder = BertModel.from_pretrained(MODEL_DIR, config=config, add_pooling_layer=False,
                                                      local_files_only=True)
        # Classification head: [CLS] representation -> Linear layer -> Class logits.
        # Khởi tạo head giống nhau giữa các chế độ, độc lập RNG của encoder.
        with torch.random.fork_rng():
            torch.manual_seed(SEED + 1)
            self.classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(.1)
        if mode == 'frozen':
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen':
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Input IDs -> Transformer -> Contextual representations.
        if self.mode == 'frozen':
            with torch.no_grad():
                contextual_representations = self.encoder(**inputs).last_hidden_state
        else:
            contextual_representations = self.encoder(**inputs).last_hidden_state
        # Contextual representations -> [CLS] representation.
        cls_representation = contextual_representations[:, 0]
        # [CLS] representation -> Dropout -> Linear layer -> Class logits.
        class_logits = self.classifier(self.dropout(cls_representation))
        return class_logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, probabilities, labels = 0., [], []
    for batch in loader:
        inputs = {key: value.to(DEVICE) for key, value in zip(INPUT_KEYS, batch[:3])}
        y = batch[3].to(device=DEVICE, dtype=torch.long)
        logits = model(**inputs)
        total_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        probabilities.append(logits.softmax(-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    prob, y = np.concatenate(probabilities), np.concatenate(labels)
    pred = prob.argmax(1)
    return {'loss': total_loss/len(y), 'y': y, 'pred': pred, 'prob': prob,
            'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred, zero_division=0), 'F1': f1_score(y, pred, zero_division=0)}


class RSSMonitor:
    """RSS toàn process lấy mẫu 20 ms; không phải peak bộ nhớ riêng của model."""
    def __enter__(self):
        self.process = psutil.Process()
        self.start = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


class LabExperiment:
    def __init__(self):
        configure()
        self.tokenizer = load_tokenizer()
        self.data, self.data_info = prepare_data()
        self.runs, self.encoded, self.analysis = {}, {}, {}
        rng = np.random.default_rng(SEED)
        labels = self.data['train'].label.to_numpy()
        self.class_orders = [rng.permutation(np.flatnonzero(labels == c)) for c in (0, 1)]
        self.hardware = {'device': str(DEVICE), 'processor': platform.processor(), 'logical_cpus': psutil.cpu_count(),
                         'torch_threads': torch.get_num_threads(), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'python': platform.python_version()}

    def subset(self, fraction):
        n = round(len(self.data['train']) * fraction)
        n0 = round(n * len(self.class_orders[0]) / len(self.data['train']))
        return np.sort(np.concatenate([self.class_orders[0][:n0], self.class_orders[1][:n-n0]]))

    def dataset(self, name, length):
        key = (name, length)
        if key not in self.encoded:
            frame = self.data[name]
            encoded = self.tokenizer(frame.text.tolist(), padding='max_length', truncation=True,
                                     max_length=length, return_tensors='pt')
            self.encoded[key] = TensorDataset(*(encoded[k] for k in INPUT_KEYS),
                                               torch.tensor(frame.label.to_numpy(), dtype=torch.long))
        return self.encoded[key]

    def run(self, mode, fraction=1., length=BASE_LENGTH):
        name = f'{mode}-n{round(fraction*100)}-T{length}'
        if name in self.runs:
            return self.runs[name]
        configure()
        chosen = self.subset(fraction)
        train_ds = Subset(self.dataset('train', length), chosen.tolist())
        valid_ds, test_ds = self.dataset('validation', length), self.dataset('test', length)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  generator=torch.Generator().manual_seed(SEED))
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        model = PretrainedSentimentClassifier(mode).to(DEVICE)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=.01)
        torch.manual_seed(SEED)
        history, best_loss, best_state, best_epoch = [], float('inf'), None, 0
        started = time.perf_counter()
        with RSSMonitor() as memory:
            for epoch in range(1, EPOCHS+1):
                model.train()
                loss_sum = 0.
                for batch in train_loader:
                    inputs = {k: v.to(DEVICE) for k, v in zip(INPUT_KEYS, batch[:3])}
                    labels = batch[3].to(device=DEVICE, dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(**inputs), labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    loss_sum += loss.item()*len(labels)
                validation = evaluate(model, valid_loader)
                history.append({'epoch': epoch, 'train_loss': loss_sum/len(train_ds),
                                'validation_loss': validation['loss'], 'validation_accuracy': validation['Accuracy']})
                if validation['loss'] < best_loss:
                    best_loss, best_epoch, best_state = validation['loss'], epoch, copy.deepcopy(model.state_dict())
                print(f'{name} | epoch {epoch}/{EPOCHS} | train={history[-1]["train_loss"]:.4f} | valid={validation["loss"]:.4f}', flush=True)
        elapsed = time.perf_counter()-started
        model.load_state_dict(best_state)
        evaluation = evaluate(model, test_loader)
        checkpoint = CACHE / f'{name}.pt'
        torch.save(best_state, checkpoint)
        row = {'Run': name, 'Model': MODEL_ID, 'Revision': REVISION, 'Mode': mode, 'Domain': 'IMDb',
               'Train fraction': fraction, 'Train samples': len(train_ds), 'Validation samples': len(valid_ds),
               'Test samples': len(test_ds), 'Max length': length, 'Batch size': BATCH_SIZE,
               'Learning rate': LR, 'Epochs': EPOCHS, 'Optimizer': 'AdamW', 'Weight decay': .01, 'Seed': SEED,
               'Parameters': sum(p.numel() for p in model.parameters()),
               'Trainable parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'Training time (s)': elapsed, 'CPU RSS baseline (MiB)': memory.start/2**20,
               'CPU RSS peak sampled (MiB)': memory.peak/2**20, 'CPU RSS increase (MiB)': (memory.peak-memory.start)/2**20,
               'Best epoch': best_epoch, 'Train subset SHA256': fingerprint(self.data['train'].iloc[chosen].source_row.tolist()),
               'Validation SHA256': self.data_info['split_hashes']['validation'], 'Test SHA256': self.data_info['split_hashes']['test'],
               **{metric: evaluation[metric] for metric in ('Accuracy', 'Precision', 'Recall', 'F1')}}
        self.runs[name] = {'row': row, 'history': history, 'checkpoint': str(checkpoint),
                           **{key: evaluation[key] for key in ('y', 'pred', 'prob')}}
        self.save()
        del model, optimizer, best_state
        gc.collect()
        return self.runs[name]

    def load_model(self, run):
        # Khởi tạo cấu trúc random rồi nạp toàn bộ checkpoint đã train.
        model = PretrainedSentimentClassifier('scratch').to(DEVICE)
        model.load_state_dict(torch.load(run['checkpoint'], map_location=DEVICE, weights_only=True))
        return model.eval()

    def domain_test(self, run):
        raw_lines = (CACHE / 'amazon_cells_labelled.txt').read_text(encoding='utf-8').splitlines()
        rows = []
        for index, line in enumerate(raw_lines):
            text, label = line.rsplit('\t', 1)
            rows.append({'text': clean_text(text), 'label': int(label), 'source_row': index})
        frame = pd.DataFrame(rows)
        raw_count = len(frame)
        frame['key'] = frame.text.str.lower()
        conflicts = frame.groupby('key').label.nunique()
        frame = frame[~frame.key.isin(conflicts[conflicts > 1].index)].drop_duplicates('key')
        imdb_keys = set(pd.concat([self.data[name] for name in ('train', 'validation', 'test')]).text.str.lower())
        frame = frame[~frame.key.isin(imdb_keys)].reset_index(drop=True)
        self.data['amazon'] = frame
        loader = DataLoader(self.dataset('amazon', run['row']['Max length']), batch_size=BATCH_SIZE)
        model = self.load_model(run)
        evaluation = evaluate(model, loader)
        result = {'Domain': 'Amazon products (UCI)', 'Samples': len(frame), 'Raw samples': raw_count,
                  'Removed duplicates/conflicts/overlap': raw_count-len(frame),
                  **{m: evaluation[m] for m in ('Accuracy', 'Precision', 'Recall', 'F1')}}
        self.analysis['domain'] = result
        self.analysis['domain_predictions'] = {k: evaluation[k] for k in ('y', 'pred', 'prob')}
        del model
        gc.collect()
        self.save()
        return result

    def save(self):
        if self.runs:
            rows = [run['row'] for run in self.runs.values()]
            if 'domain' in self.analysis:
                domain = self.analysis['domain']
                row = self.runs['finetuned-n100-T128']['row'].copy()
                row.update({'Run': 'finetuned-n100-T128-amazon', 'Domain': domain['Domain'], 'Test samples': domain['Samples'],
                            'Test SHA256': fingerprint(self.data['amazon'].text.tolist()),
                            **{m: domain[m] for m in ('Accuracy', 'Precision', 'Recall', 'F1')}})
                rows.append(row)
            pd.DataFrame(rows).to_csv(LAB_DIR / 'results.csv', index=False)
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value))
        info = {'data': self.data_info, 'hardware': self.hardware,
                'runs': self.runs, 'analysis': self.analysis}
        (CACHE / 'evidence.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')


@torch.no_grad()
def predict_text(model, tokenizer, text, max_length=BASE_LENGTH):
    # Input text -> Tokenizer -> Input IDs (và mask).
    tokenized_inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    model_inputs = {key: value.to(DEVICE) for key, value in tokenized_inputs.items()}
    class_logits = model(**model_inputs)
    # Class logits -> Softmax -> Prediction.
    probabilities = class_logits.softmax(-1)[0].cpu().numpy()
    prediction = int(probabilities.argmax())
    return {'Text': text, 'Predicted': prediction, 'Label': 'positive' if prediction else 'negative',
            'P(positive)': float(probabilities[1]), 'Confidence': float(probabilities.max()),
            'Visible text': tokenizer.decode(tokenized_inputs['input_ids'][0], skip_special_tokens=True)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='I really enjoyed this movie.')
    args = parser.parse_args()
    configure()
    tokenizer = load_tokenizer()
    checkpoint = CACHE / 'finetuned-n100-T128.pt'
    if not checkpoint.exists():
        raise SystemExit('Chạy mục 14 trong fine_tuning.ipynb trước để tạo checkpoint sentiment.')
    classifier = PretrainedSentimentClassifier('scratch')
    classifier.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    classifier.eval()
    print(json.dumps(predict_text(classifier, tokenizer, args.text), ensure_ascii=False, indent=2))
