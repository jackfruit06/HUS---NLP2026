# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

---
# LAB 09 — Named Entity Recognition and Information Extraction

**Chủ đề:** From Sequence Classification to Token Classification  
**Dataset:** CoNLL-2003 English; 30 câu news-style tự biên soạn để đánh giá IE.  
**Kết nối LAB 08:** Thay head đọc [CLS] bằng head dự đoán BIO tại mỗi vị trí.

## Xem notebook

- [Tokenization và alignment](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab9/tokenizer_alignment.ipynb)
- [NER fine-tuning và IE](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab9/ner_finetuning.ipynb)

## Cấu trúc thư mục

```text
lab9/
├── README.md                      <- file này
├── calculations.md                <- BIO, entity boundary và tensor shapes
├── prediction.md                  <- 3 predictions trước experiment
├── baseline_ner.py                <- token features, alignment, BERT token classifier, metrics
├── tokenizer_alignment.ipynb      <- mục 1–12, WordPiece và word_ids/labels
├── ner_finetuning.ipynb           <- mục 13–27, 29, 32; experiments và câu trả lời
├── results.csv                    <- baseline và Transformer, entity-level metrics
├── entity_metrics.csv             <- Precision/Recall/F1 theo PER/ORG/LOC/MISC
├── entity_length_analysis.png     <- F1 theo độ dài entity
├── confusion_matrix.png           <- confusion matrix token BIO
├── error_analysis.md              <- 5 FP, 5 FN, 5 boundary errors và context
├── information_extraction.py      <- structured extraction và gold IE 30 câu
└── reflection.md                  <- reflection 300–500 từ
```

## Dữ liệu dùng

[CoNLL-2003](https://huggingface.co/datasets/tomaarsen/conll2003), giữ split gốc: 14041 train, 3250 validation, 3453 test. Dùng toàn bộ train, không fit vocabulary/features trên test. Snapshot revision và SHA256 từng parquet nằm trong output mục 29. BIO được chuẩn hóa về BIO2. Chọn checkpoint theo validation entity F1; test chỉ báo cáo. IE dùng gold gán tay trong module, không dùng làm training hoặc chọn checkpoint.

## Kết quả chính

| Model | Precision | Recall | F1 | Training time (s) |
| --- | --- | --- | --- | --- |
| Baseline | 0.6737 | 0.7096 | 0.6912 | 6.5490 |
| Pretrained Transformer | 0.7707 | 0.7999 | 0.7851 | 114.7377 |

Các metric trên là micro entity-level exact span+type; confusion matrix là token-level bổ trợ. BERT-Tiny: 2 layers, hidden 128, 2 heads, head 9 nhãn; không phải BERT-base. Alignment A, max length 128, chunk ở ranh giới word và ghép lại đầy đủ; batch 32, AdamW lr=3e-4, 4 epochs, seed 42, CPU 4 threads. Một seed chưa đo độ biến thiên. Baseline dùng đặc trưng chữ hoa, BERT-Tiny uncased không giữ chữ hoa.

IE trên 30 câu tự biên soạn: Precision=0.8681, Recall=0.8778, F1=0.8729. Đây là tập theo mẫu có kiểm soát, không đại diện mọi tin tức.

## Cách chạy

```bash
pip install torch transformers==5.22.2 numpy pandas scikit-learn matplotlib seaborn nbformat ipython jupyter requests
pip install pyarrow==18.1.0
cd lab9
python baseline_ner.py
python information_extraction.py --text "Barack Obama visited Paris."
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 ner_finetuning.ipynb
```

Notebook đọc kết quả/checkpoint đã lưu; khi chưa có, `run()` huấn luyện baseline rồi Transformer. Module CLI `baseline_ner.py` luôn huấn luyện lại. Cache `.lab9_cache/` ở thư mục gốc chứa dữ liệu, checkpoint và evidence; lần đầu cần mạng. `pyarrow` chỉ cần để chuyển parquet sang JSON lúc đầu; workspace hiện tại có bản riêng tại `.lab9_cache/deps`. Mục 29 ghi cấu hình/phần mềm, hash và thời gian. Khi chạy lại training, nhận xét cần được đối chiếu với kết quả mới.

Tài liệu: [Token classification và alignment](https://huggingface.co/docs/transformers/tasks/token_classification), [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/), [BERT-Tiny checkpoint](https://huggingface.co/prajjwal1/bert-tiny).
