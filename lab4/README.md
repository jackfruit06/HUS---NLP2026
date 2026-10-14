# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification.

---
# LAB 04 — Text Classification

**Application (mục 23):** Option A — Sentiment classification (positive / negative) trên IMDb, kèm cách dùng với ngưỡng tự tin.

## Xem notebook

Nếu GitHub không hiển thị được `classification.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab4/classification.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab4/classification.ipynb)

## Cấu trúc thư mục

```
lab4/
├── README.md             <- file này
├── Prediction.pdf         <- prediction trước khi chạy experiment (Prediction 1-5, mục 10)
├── Calculations.pdf       <- bài tính tay: TF-IDF, z = w^T*x+b và sigmoid, Accuracy/Precision/Recall/F1, average embedding
├── preprocessing.py      <- Core implementation: load/clean IMDb, split, TF-IDF, average Word2Vec, Logistic Regression / Linear SVM, metrics, top features, explain_prediction
├── classification.ipynb  <- Experiment 1-5 (mục 11-22), Application (mục 23, Option A), experiment bổ sung kiểm chứng Prediction 3, 4, 5
├── results.csv           <- Kết quả định lượng (mean của 3 seed) của các experiment
├── confusion_matrix.png  <- Confusion matrix của 3 model (test, seed 42)
├── Error analysis.pdf     <- phân tích 3 review đúng, 4 review sai (mục 21)

```

Dữ liệu dùng: `../dataset/IMDB Dataset.csv` — 50,000 review phim gán nhãn positive/negative (cân bằng 25,000/25,000),
tải tại [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
Bỏ 418 review trùng lặp (còn 49,578), bỏ thẻ `<br />`, chia 70/15/15 (train/validation/test, stratify theo label) với 3 seed `42, 123, 456`.
Mọi bước fit (TF-IDF, Word2Vec, StandardScaler, chọn `C`) chỉ dùng **train** (và **validation** để chọn `C`); test chỉ dùng để báo cáo.

## Cách chạy

```bash
cd lab4
pip install numpy scipy pandas scikit-learn gensim matplotlib jupyter
jupyter nbconvert --to notebook --execute --inplace classification.ipynb   # chạy lại toàn bộ experiment (~30 phút trên 1 CPU, chủ yếu là train Word2Vec), ghi kết quả ra results.csv, confusion_matrix.png và figures/
```

## Kết quả chính

| Representation | Classifier | Accuracy | F1 | Training time |
|---|---|---:|---:|---:|
| TF-IDF | Logistic Regression | 0.8989 | 0.9005 | khoảng 6.8s |
| Avg Word2Vec | Logistic Regression | 0.8691 | 0.8703 | khoảng 250s |
| TF-IDF | Linear SVM | 0.8998 | 0.9017 | khoảng 6.3s |
| TF-IDF unigram + bigram | Logistic Regression | 0.9164 | 0.9176 | khoảng 28.0s |


