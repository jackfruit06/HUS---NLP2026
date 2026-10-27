# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification, sequence modeling và attention.

---
# LAB 06 — Attention Mechanism

**Chủ đề:** From Recurrent State to Context-Aware Representation  
**Dataset:** IMDb Movie Reviews (50,000 reviews, sentiment classification positive/negative)  
**Kết nối với LAB 05:** Sử dụng lại IMDb dataset nhưng thay đổi từ final hidden state của LSTM sang context representation từ Self-Attention.

## Xem notebook

Nếu GitHub không hiển thị được `self_attention.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab06/self_attention.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab06/self_attention.ipynb)

## Cấu trúc thư mục

```
lab06/
├── README.md                <- file này
├── calculations.md          <- tính tay: scores, softmax, context, scaling, self-attention
├── prediction.md            <- 5 predictions trước khi chạy experiment
├── attention_numpy.py       <- Scaled Dot-Product Attention (NumPy) + numerical stability + verification
├── self_attention.ipynb     <- Experiment 1-5 + Application + Error Analysis + trả lời câu hỏi
├── results.csv              <- Kết quả định lượng của các experiments và models (test set)
├── attention_heatmap.png    <- Attention matrix NumPy giả lập của 2 câu trong đề
├── confusion_matrix.png     <- Confusion matrix của 2 models (LSTM, Attention)
├── error_analysis.md        <- Phân tích 3 correct + 3 incorrect predictions của Attention
└── reflection.md            <- Reflection: predictions, surprising results,
                                Attention→Transformer motivation
```

## Dữ liệu dùng

`../dataset/IMDB Dataset.csv` — 50,000 review phim gán nhãn positive/negative (25,000/25,000).  
Tiền xử lý: bỏ thẻ `<br />`, chuẩn hóa khoảng trắng, bỏ 422 duplicate theo text đã clean (còn 49,578 reviews), chia 70/15/15 (train/validation/test, stratify theo label, seed=42).  
Sử dụng subset như LAB 05: 10,000 train, 2,000 validation, 2,000 test; vocabulary có 33,514 tokens, `min_freq=2`, `max_length=100`.  
**Vocabulary chỉ fit trên train set; checkpoint chọn theo validation loss; test chỉ dùng để đánh giá.**

## Kết quả chính

| Model | Representation | Accuracy | F1 | Train time |
|---|---|---:|---:|---:|
| LSTM + Embedding | Final hidden state (dim=64) | 0.7105 | 0.7384 | ~17,35 s |
| Self-Attention + Embedding | Context + masked mean pooling (dim=64) | 0.7775 | 0.7647 | ~17,07 s |

Hai model được huấn luyện trên cùng dữ liệu với `epochs=5`, `batch_size=128`, `lr=0.002`, seed=42; đều chọn checkpoint epoch 3 theo validation loss. Thời gian đo trên CPU 4 threads, gồm training và validation. Phân tích kết quả và giới hạn được trình bày trong notebook, `error_analysis.md` và `reflection.md`.

## Cách chạy

```bash
cd lab06

# 1. Kiểm tra Scaled Dot-Product Attention
python attention_numpy.py

# 2. Chạy toàn bộ experiments và lưu output
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 self_attention.ipynb

# Hoặc mở Jupyter và chạy từng cell
jupyter notebook self_attention.ipynb
```

**Dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch ipython jupyter nbconvert
```
