# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification, và sequence modeling.

---
# LAB 05 — Sequence Models: RNN, LSTM and GRU

**Chủ đề:** From Static Representations to Sequential Modeling  
**Dataset:** IMDb Movie Reviews (50,000 reviews, sentiment classification positive/negative)  
**Kết nối với LAB 04:** Sử dụng lại IMDb dataset nhưng thay đổi từ bag-of-words (TF-IDF) sang sequence modeling (RNN/LSTM/GRU).

## Xem notebook

Nếu GitHub không hiển thị được `sequence_classification.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab5/sequence_classification.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab5/sequence_classification.ipynb)

## Cấu trúc thư mục

```
lab05/
├── README.md                    <- file này
├── calculations.md              <- tính tay: Bài 1 RNN forward, LSTM gate, sequence order
├── prediction.md                <- 5 predictions trước khi chạy experiment
├── rnn_cell.py                  <- SimpleRNNCell (NumPy) + tiền xử lý + verification vs PyTorch
├── sequence_classification.ipynb <- Experiment 1-6 + Application + Error Analysis
├── results.csv                  <- Kết quả định lượng của tất cả models (test set)
├── learning_curves.png          <- Train & validation loss/accuracy qua epoch (RNN/LSTM/GRU)
├── confusion_matrix.png         <- Confusion matrix của 3 models (RNN, LSTM, GRU)
├── error_analysis.md            <- Phân tích 3 correct + 4 incorrect predictions của LSTM
└── reflection.md                <- Reflection: wrong predictions, surprising results,
                                    RNN→Attention motivation
```


## Dữ liệu dùng

`../dataset/IMDB Dataset.csv` — 50,000 review phim gán nhãn positive/negative (25,000/25,000).  
Tiền xử lý: bỏ 418 duplicate, bỏ thẻ `<br />`, chia 70/15/15 (train/validation/test, stratify theo label, seed=42).  
**Mọi fit (Vocabulary, TF-IDF) chỉ trên train set** → không có data leakage.

## Kết quả chính

| Model | Representation | Accuracy | F1 | Train time |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | Sparse | ~0.8815 | ~0.8837 | ~0,12s |
| RNN + Embedding | Dense (dim=64) | ~0.5095 | ~0.4717 | ~22,6 s |
| LSTM + Embedding | Dense (dim=64) | ~0.6535 | ~0.5818 | ~23,9 s |
| GRU + Embedding | Dense (dim=64) | ~0.7145 | ~0.7420 | ~36,1 s |


## Cách chạy

```bash
cd lab5

# 1. Kiểm tra SimpleRNNCell
python rnn_cell.py

# 2. Chạy toàn bộ experiments (mất ~45-60 phút trên CPU)
jupyter nbconvert --to notebook --execute --inplace sequence_classification.ipynb

# Hoặc mở Jupyter và chạy từng cell
jupyter notebook sequence_classification.ipynb
```

**Dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter nbconvert
```
