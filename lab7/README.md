# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) từ xử lý văn bản, word embeddings, classification, sequence modeling, attention đến Transformer.

---
# LAB 07 — Transformer Architecture

**Chủ đề:** From Attention to Transformer  
**Dataset:** IMDb Movie Reviews (50,000 reviews, sentiment positive/negative)  
**Kết nối với LAB 06:** Ghép multi-head self-attention với positional encoding, FFN, residual và LayerNorm thành Transformer Encoder block; dùng cho text classification.

## Xem notebook

Nếu GitHub không hiển thị được `transformer_classification.ipynb`, xem notebook (kèm output) tại:

- [Xem trên nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab7/transformer_classification.ipynb)
- [Mở trên Google Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab7/transformer_classification.ipynb)

## Cấu trúc thư mục

```text
lab7/
├── README.md                       <- file này
├── calculations.md                 <- tính attention, multi-head, PE và FFN
├── prediction.md                   <- 6 predictions trước experiment
├── transformer_block.py            <- multi-head attention và encoder block tự ghép
├── transformer_classification.ipynb <- mục 1–27 theo đề, output và câu trả lời Markdown
├── results.csv                     <- kết quả của 9 cấu hình được huấn luyện
├── attention_heads.png             <- attention matrix của 4 heads
├── error_analysis.md               <- phân tích 3 correct + 3 incorrect
└── reflection.md                   <- đối chiếu prediction và kết quả, giới hạn
```

## Dữ liệu dùng

`../dataset/IMDB Dataset.csv` — 50,000 review, hai lớp positive/negative. Clean thẻ `<br />` và khoảng trắng, bỏ 422 review trùng sau clean, còn 49,578. Dùng cùng cách chia và subset của LAB 06: stratified 70/15/15, seed=42; 10,000 train, 2,000 validation, 2,000 test. Vocabulary 33,514 tokens, `min_freq=2`, chỉ fit trên train.

**Checkpoint chọn bằng validation loss; test chỉ dùng để báo cáo.** Mọi mô hình dùng embedding học từ đầu, dimension 128, batch 64, Adam lr=0.002, 5 epochs. Transformer có 1 block, d_ff=256, dropout block 0.1; trước classifier dropout 0.3 cho cả LSTM và Transformer. Độ dài cơ sở 64; riêng experiment sequence length thử 32/64/128/256. So sánh trong LAB 07 dùng cùng split/vocabulary; không so trực tiếp số đo với LAB 05 có tiền xử lý/cấu hình khác.

## Kết quả chính

| Model | Representation | Accuracy | F1 | Train time |
|---|---|---:|---:|---:|
| LSTM + Embedding | Final valid hidden state (dim=128) | 0.7145 | 0.7253 | 27.89 s |
| Transformer + Embedding | 4 heads + PE + masked mean (dim=128) | 0.7550 | 0.7567 | 48.22 s |

Thời gian đo trên CPU 4 threads, gồm training và validation, không gồm preprocessing/test. Đây là một seed, không phải kết luận Transformer luôn tốt hơn hoặc nhanh hơn LSTM. `results.csv` có thêm thử nghiệm số head, độ dài và pooling; cấu hình trùng được tái sử dụng.

## Cách chạy

Đặt dataset tại `dataset/IMDB Dataset.csv` ở thư mục gốc của repository, sau đó:

```bash
cd lab7

# Chạy toàn bộ notebook và lưu output
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 transformer_classification.ipynb

# Hoặc mở notebook và Run All
jupyter notebook transformer_classification.ipynb
```

Notebook cũng xác định đúng đường dẫn khi chạy từ thư mục gốc repository. Chạy lại sẽ huấn luyện các cấu hình và cập nhật `results.csv`, `attention_heads.png`; thời gian phụ thuộc máy. Nhận xét Markdown ghi số đo của lần chạy đã lưu, nên cần cập nhật nếu kết quả mới khác.

**Dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch ipython jupyter nbconvert
```
