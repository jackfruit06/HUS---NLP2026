# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification.

---
# LAB 01 — From Text Processing to Search

## Cấu trúc thư mục

```
lab1/
├── README.md            <- file này
├── calculations.md      <- bài tính tay (Exercise 1-6)
├── prediction.md         <- prediction trước khi mở corpus 30K (Prediction 1-3)
├── preprocessing.py      <- 3 pipeline tokenize (A: minimal, B: normalized, C: extended)
├── implementation.py     <- Core TF-IDF và cosine similarity.
├── experiments.ipynb     <- Các experiment trên corpus 30K.
├── results.csv           <- Kết quả retrieval/evaluation.
├── reflection.md         <- reflection cuối bài (6 câu hỏi)
└── dataset/
    └── 30k_documents.json  <- corpus 30,000 documents (JSON Lines: text, timestamp, url)
```

Dữ liệu dùng: `D:\NLP\lab1\dataset\30k_documents.json` (30,000 documents, định dạng JSON Lines,
mỗi dòng có `text`, `timestamp`, `url`).

## Cách chạy

```bash
cd lab1
python implementation.py          # chạy unit test Part E trên corpus toy (cat/dog/fish)
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb   # chạy lại toàn bộ Part D/F/G/H/I/J
```


