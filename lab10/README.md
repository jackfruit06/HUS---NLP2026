# Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)

---
# LAB 10 — Extractive Question Answering

**Chủ đề:** From Token Classification to Span Prediction  
**Dataset:** SQuAD 1.1  
**Kết nối LAB 09:** Đổi BIO head thành hai classifier chọn start/end answer span.

## Xem notebook

- [Tokenization và offset alignment](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab10/tokenizer_alignment.ipynb)
- [QA fine-tuning và Document QA](https://nbviewer.org/github/jackfruit06/HUS---NLP2026/blob/master/lab10/qa_finetuning.ipynb)

## Cấu trúc thư mục

```text
lab10/
├── README.md                      <- file này
├── calculations.md                <- start/end, loss, character span và QA F1
├── prediction.md                  <- 4 predictions trước experiment
├── tokenizer_alignment.ipynb      <- mục 1–13: tokenization, offsets và bài tính
├── qa_model.py                    <- baseline, alignment, span head, training và metrics
├── qa_finetuning.ipynb            <- mục 14–28, 30, 31, 34 và câu trả lời
├── results.csv                    <- baseline và ba cấu hình context length
├── learning_curves.png            <- training/validation loss và validation F1
├── em_f1_analysis.png             <- so sánh EM/F1 và mức overlap từng câu
├── answer_length_analysis.png     <- EM/F1 theo độ dài gold answer
├── error_analysis.md              <- 5 wrong start, 5 wrong end, 5 wrong span
├── document_qa.py                 <- QA trên context nhập trực tiếp
└── reflection.md                  <- so sánh backbone và head LAB 08–10
```

## Dữ liệu dùng

[SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/): dùng subset 3.000 câu hỏi từ official train; official dev được chia theo context rồi lấy 300 validation và 600 test, seed=42. Không trùng context giữa ba tập. Test của bài là holdout từ dev, không phải hidden official test. Cùng IDs được dùng cho mọi cấu hình. Dữ liệu, checkpoint và evidence lưu ở `.lab10_cache/`; hash JSON và split IDs được in ở mục 30.

## Kết quả chính

| Model | EM | F1 |
| --- | --- | --- |
| Baseline | 7.6667 | 16.8053 |
| Fine-tuned Transformer | 16.3333 | 27.5675 |

EM/F1 theo thang 0–100, normalization theo SQuAD, chấm max qua gold references. Cấu hình cơ sở T=256 được cố định cho phân tích/application. BERT-Tiny có 2 layers, hidden 128, 2 heads; encoder pretrained chung được fine-tune cùng span head mới, không phải model đã fine-tune QA từ trước.

| Max length | EM | F1 | Training time |
| --- | --- | --- | --- |
| 128 | 9.8333 | 21.3354 | 119.8091 |
| 256 | 16.3333 | 27.5675 | 121.3028 |
| 384 | 14.6667 | 27.2928 | 157.6939 |

Training time tính bằng giây, gồm train + validation và lưu best state trong RAM; không gồm download/tokenization/test/pretraining. AdamW lr=3e-4, batch=16, 3 epochs, seed=42, CPU 4 threads. Question tối đa 48 subword; stride=64; dynamic padding; decoder top-20 start/end, max answer=64 subword, chỉ chọn context positions và end≥start. Checkpoint theo validation F1 cao nhất. Độ dài làm đổi số window/steps nên đây không phải so sánh cùng compute budget. Chỉ một seed và subset CPU, chưa tối ưu hyperparameters.

## Cách chạy

```bash
pip install torch transformers==5.22.2 numpy pandas matplotlib seaborn requests nbformat ipython jupyter
cd lab10
python qa_model.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 qa_finetuning.ipynb
python document_qa.py --question "What is the capital of France?" --context "The capital of France is Paris."
```

Lần đầu tải dữ liệu/model công khai; các lần sau dùng cache. Notebook dùng kết quả training đã lưu, còn `qa_model.py` chạy lại ba cấu hình. Các phép phân tích context và error được triển khai trong module, notebook tính lại từ prediction khi cần. Nhận xét Markdown ghi số đo của lần chạy đã lưu. Core dùng SQuAD 1.1, luôn chọn span; chưa đánh giá khả năng từ chối no-answer. Document QA không retrieval; LAB 11 mới truy hồi tài liệu, chuẩn bị cho RAG ở LAB 12.

Tài liệu: [SQuAD paper](https://arxiv.org/abs/1606.05250), [Hugging Face QA và offset/window mapping](https://huggingface.co/docs/course/main/chapter7/7), [Official evaluation](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py).
