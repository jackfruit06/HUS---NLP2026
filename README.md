# Natural Language Processing Laboratory Assignments

> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

---

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) — HUS NLP 2026. Các lab đi từ xử lý và tìm kiếm văn bản, mô hình ngôn ngữ thống kê (N-gram), biểu diễn từ dạng vector (word embeddings), phân loại văn bản đến mô hình hóa chuỗi với RNN, LSTM và GRU.

## Danh sách bài thực hành

| Lab | Nội dung | Kỹ thuật chính | Notebook |
|-----|----------|----------------|----------|
| [lab1](lab1/) | From Text Processing to Search | Tokenization, TF-IDF, cosine similarity, document retrieval | [experiments.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab1/experiments.ipynb) |
| [lab2](lab2/) | Language Models | N-gram LM (unigram/bigram/trigram), MLE, Laplace smoothing, perplexity, next-word prediction | [experiments.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab2/experiments.ipynb) |
| [lab3](lab3/) | Word Representations and Embeddings | Co-occurrence matrix, Word2Vec (Skip-gram), word similarity, word analogy, semantic search | [word_embedding.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab3/word_embedding.ipynb) |
| [lab4](lab4/) | Text Classification | TF-IDF, Avg Word2Vec, Logistic Regression, Linear SVM, sentiment classification, confusion matrix, error analysis | [classification.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab4/classification.ipynb) |
| [lab5](lab5/) | Sequence Models: RNN, LSTM and GRU | SimpleRNNCell (NumPy), sequence padding/truncation, sentiment classification, long-range dependency, error analysis | [sequence_classification.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab5/sequence_classification.ipynb) |

## Cấu trúc kho lưu trữ

```text
HUS---NLP2026/
├── README.md
├── dataset/
├── lab1/   <- LAB 01 — From Text Processing to Search
├── lab2/   <- LAB 02 — Language Models
├── lab3/   <- LAB 03 — Word Representations and Embeddings
├── lab4/   <- LAB 04 — Text Classification
└── lab5/   <- LAB 05 — Sequence Models: RNN, LSTM and GRU
```

Mỗi lab có `README.md` riêng mô tả cấu trúc thư mục, dữ liệu và cách chạy. Các phần chung của mỗi lab:

- `prediction` — dự đoán trước khi chạy experiment;
- `calculations` — bài tính tay;
- file `.py` — phần core tự cài đặt;
- notebook — các experiment, kèm output;
- `results.csv` — kết quả định lượng;
- `error analysis`, `reflection` — phân tích lỗi và tổng kết.

## Dữ liệu

- `/dataset/30k_documents.json` — corpus 30,000 documents (JSON Lines: `text`, `timestamp`, `url`), dùng chung cho các lab liên quan.
- `/dataset/IMDB Dataset.csv` — 50,000 review phim có nhãn sentiment, dùng trong Lab 4 và Lab 5 để phân loại cảm xúc.

## Môi trường

- Python 3.12
- Thư viện tùy theo lab: `numpy`, `scipy`, `pandas`, `scikit-learn`, `gensim`, `matplotlib`, `seaborn`, `torch`, `jupyter`, `nbconvert`

```bash
pip install numpy scipy pandas scikit-learn gensim matplotlib seaborn torch jupyter nbconvert
```
