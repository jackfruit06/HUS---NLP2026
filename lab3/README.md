# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification.

---
# LAB 03 — Word Representations and Embeddings

## Xem notebook

Nếu GitHub không hiển thị được `word_embedding.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab3/word_embedding.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab3/word_embedding.ipynb)

## Cấu trúc thư mục

```
lab3/
├── README.md             <- file này
├── prediction.md         <- prediction trước khi chạy experiment
├── calculations.md       <- bài tính tay (training pairs CBOW/Skip-gram, analogy với vector giả định)
├── cooccurrence.py       <- Core implementation: tokenize, vocabulary, co-occurrence matrix (sparse), cosine, most_similar
├── word_embedding.ipynb  <- Experiment 1-4, word similarity, analogy, semantic search, error analysis, polysemy
├── results.csv           <- Kết quả định lượng của Experiment 1, 3, 4.
├── error_analysis.md     <- phân tích 3 similarity đúng, 3 similarity sai/bất ngờ
├── reflection.md         <- reflection cuối bài (từ Word2Vec đến Transformer)
└── IMDB Dataset.csv      <- 50,000 review phim cho downstream task 
```

Dữ liệu dùng:
- Corpus: 10,000 documents đầu tiên của `../lab1/dataset/30k_documents.json` (định dạng JSON Lines,
  mỗi dòng có `text`, `timestamp`, `url`), dùng cho cả co-occurrence matrix và Word2Vec.
- Downstream task (Experiment 4): `IMDB Dataset.csv` — 50,000 review phim gán nhãn positive/negative,
  tải tại [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) và đặt vào `lab3/`.
- Evaluation: Google analogy (`questions-words.txt`) có sẵn trong `gensim`, dùng ở Experiment 4.

## Cách chạy

```bash
cd lab3
pip install numpy scipy pandas scikit-learn gensim jupyter
jupyter nbconvert --to notebook --execute --inplace word_embedding.ipynb   # chạy lại toàn bộ experiment (~15-20 phút), ghi kết quả ra results.csv
```
