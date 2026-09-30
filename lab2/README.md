# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification.

---
# LAB 02 — Language Models

## Xem notebook

Nếu GitHub không hiển thị được `experiments.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/main/lab2/experiments.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/main/lab2/experiments.ipynb)

## Cấu trúc thư mục

```
lab2/
├── README.md            <- file này
├── Prediction.pdf        <- prediction trước khi chạy experiment (Prediction 1-5)
├── Caculation.pdf        <- bài tính tay (log probability, perplexity)
├── preprocessing.py      <- đọc corpus, tách câu, tokenize, chia train/valid/test
├── ngram_lm.py           <- Core N-gram Language Model (unigram/bigram/trigram, MLE/Laplace)
├── experiments.ipynb     <- Các experiment trên corpus 30K.
├── results.csv           <- Kết quả perplexity và next-word prediction.
├── Error Analysis.pdf    <- phân tích 2 prediction đúng, 2 prediction sai
├── Reflection.pdf        <- reflection cuối bài
└── reflection.md
```

Dữ liệu dùng: `../lab1/dataset/30k_documents.json` (30,000 documents, định dạng JSON Lines,
mỗi dòng có `text`, `timestamp`, `url`), chia 80/10/10 theo document (seed = 42).

## Cách chạy

```bash
cd lab2
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb   # chạy lại toàn bộ experiment, ghi kết quả ra results.csv
```
