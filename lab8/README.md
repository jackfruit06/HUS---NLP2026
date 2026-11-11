# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

## Tổng quan dự án

Tổng hợp các bài thực hành **Xử lý Ngôn ngữ Tự nhiên**, từ biểu diễn văn bản, classification, sequence models và Transformer đến pretrained representations và fine-tuning.

---
# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

**Chủ đề:** From Training a Transformer to Using a Pretrained Language Model  
**Dataset:** IMDb Movie Reviews; Amazon product-review sentences để kiểm tra domain shift.  
**Kết nối LAB 07:** Giữ kiến trúc Transformer nhưng so sánh khởi tạo random, frozen pretrained encoder và fine-tuning.

## Xem notebook

- Tokenization, mục 1–12: [nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab8/tokenization.ipynb) · [Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab8/tokenization.ipynb).
- Fine-tuning, mục 13–23 và 28–29: [nbviewer](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab8/fine_tuning.ipynb) · [Colab](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab8/fine_tuning.ipynb).

Các câu trả lời và nhận xét được trình bày tại mục tương ứng trong hai notebook.

## Cấu trúc thư mục

```text
lab8/
├── README.md                    <- file này
├── Calculation.pdf              <- tính tay MLM loss và bài tập tokenization
├── Prediction.pdf               <- 3 predictions trước khi chạy experiment
├── tokenization.ipynb           <- WordPiece, special tokens, attention mask và token IDs
├── pretrained_classification.py <- BERT-Tiny classifier, training/evaluation và inference
├── fine_tuning.ipynb            <- Experiment 1–5, Application, Error/Confidence Analysis
├── results.csv                  <- kết quả 11 cấu hình training và đánh giá khác domain
├── data_efficiency.png          <- Accuracy/F1 theo lượng dữ liệu của scratch và fine-tuned
├── confusion_matrix.png         <- confusion matrix của fine-tuned BERT-Tiny trên IMDb
├── confidence_analysis.png      <- so sánh confidence với accuracy và phân bố dự đoán
├── Error analysist.pdf          <- phân tích lỗi và trả lời Q1–Q6
├── reflection.md                <- reflection về pretraining, fine-tuning và giới hạn
└── Reflection.pdf               <- bản viết tay reflection
```

`pretrained_classification.py` chứa model, dữ liệu, training/evaluation và inference. CSV có 11 cấu hình training và 1 dòng đánh giá Amazon dùng lại checkpoint IMDb; dòng Amazon không phải lần training mới. Ảnh data efficiency có cả scratch và fine-tuned. Error analysis gồm 3 FP, 3 FN, 2 lỗi confidence cao khác nhau; reflection tổng kết kết quả và giới hạn của mô hình.

## Dữ liệu dùng

`../dataset/IMDB Dataset.csv`: 50.000 review → 49.578 sau clean/dedup; cùng split/subset LAB 07: 10.000 train, 2.000 validation, 2.000 test, seed 42. Subset train 10/25/50/100% lồng nhau, giữ validation/test cố định. Tokenizer là vocabulary WordPiece cố định của checkpoint, không fit lại trên test. Chọn best checkpoint bằng validation loss.

Domain B: [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias (2015), DOI 10.24432/C57604, CC BY 4.0. Dùng phần Amazon 1.000 câu, còn 987 sau loại trùng/conflict/overlap; không dùng nhãn Amazon để train hay chọn model. Model và dữ liệu tải về cache `.lab08_cache/` ở thư mục gốc, ngoài thư mục nộp bài.

## Kết quả chính

| Model | Representation | Accuracy | F1 | Train time |
|---|---|---:|---:|---:|
| BERT-Tiny from scratch | Random encoder + [CLS] + Linear | 0.8245 | 0.8234 | 143.78 s |
| BERT-Tiny frozen | Pretrained [CLS], chỉ train head | 0.6870 | 0.6793 | 41.45 s |
| BERT-Tiny fine-tuned | Encoder + head đều cập nhật | 0.8180 | 0.8162 | 150.26 s |

Model: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny), revision `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`, 2 layers, hidden 128, 2 heads; không phải BERT-base. Cùng T=128, batch 64, AdamW lr=2e-4, 3 epochs, CPU 4 threads; chỉ chạy một seed. Thời gian không gồm download/tokenization/test hoặc chi phí pretraining. Bộ nhớ báo cáo là process RSS lấy mẫu 20 ms, không phải bộ nhớ riêng model. Reproducibility đầy đủ ở mục 23.

## Cách chạy

Giữ module `.py` cạnh hai notebook và đặt IMDb tại `dataset/IMDB Dataset.csv` ở thư mục gốc. Cài trong môi trường Python của notebook:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn psutil requests ipython jupyter nbconvert
pip install transformers==5.22.2 huggingface-hub==2.4.0 tokenizers==0.23.3 safetensors==0.8.0
cd lab8
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 tokenization.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 fine_tuning.ipynb
python pretrained_classification.py --text "I really enjoyed this movie."
```

Hai notebook chạy độc lập; notebook fine-tuning tạo checkpoint cho lệnh inference. Có thể mở Jupyter và Run All. Cần mạng lần đầu để tải checkpoint công khai và Amazon UCI; những lần sau dùng cache. Trong VS Code, chọn kernel Python có các phiên bản thư viện ghi bên dưới để tái lập môi trường của lần chạy đang báo cáo. Chạy lại cập nhật CSV/ảnh/checkpoint; nhận xét Markdown ghi số đo của lần chạy đã lưu nên cần đối chiếu nếu đổi cấu hình.

Môi trường đã chạy: Python 3.12.7, PyTorch 2.11.0+cpu, Transformers 5.22.2, CPU `AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD`. Tài liệu: [BERT API v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert).
