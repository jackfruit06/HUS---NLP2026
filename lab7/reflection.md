# LAB 07 — Reflection

## 1. Đối chiếu prediction

| Prediction | Đối chiếu kết quả |
|---|---|
| 1. Transformer song song hơn RNN | Kiến trúc Q/K/V không phụ thuộc state timestep trước. Thời gian T=64: Transformer 48.22s, LSTM 27.89s; mức song song không đồng nghĩa chắc chắn nhanh hơn. |
| 2. Không PE thì không có thứ tự tuyệt đối | Khi đảo `I love NLP`, embedding chỉ hoán vị hàng; thêm PE làm cùng token ở hai vị trí có đầu vào khác nhau. |
| 3. Không PE + mean không phân biệt hai hoán vị | `dog bites man` / `man bites dog`: L2 không PE 3.725e-09, có PE 0.041969. Hỗ trợ prediction trong điều kiện experiment. |
| 4. Heads có thể học pattern khác nhau | Các cặp head có sai khác attention trung bình 0.1199–0.1838 trên câu đã chọn; chưa đủ để gán vai trò ngôn ngữ. |
| 5. Nhiều heads không đảm bảo accuracy tăng | Accuracy h=1/2/4/8: 0.7570, 0.7475, 0.7550, 0.7505. Một lần chạy không đảm bảo quy luật chung. |
| 6. Transformer không luôn nhanh hơn RNN | Đã đo hai model ở T=64 và Transformer ở 4 độ dài. Không có benchmark LSTM ở mọi T để kết luận toàn bộ đường cong tốc độ; chi phí attention vẫn tăng theo T². |

## 2. Kết quả cần chú ý

Số head thay đổi nhưng số tham số vẫn **4,422,530** vì d_model cố định; d_k giảm theo 128/h. Nhiều head hơn tăng số ma trận attention chứ không tăng tổng representation dimension.

Không PE, hai câu đảo vai trò chủ thể/đối tượng có cùng mean representation dù nghĩa khác nhau. Khi thêm PE, vector khác nhau; đây là khả năng phân biệt về biểu diễn, chưa phải bằng chứng hiểu nghĩa.

Mean pooling có F1 **0.7567**, first-token có F1 **0.5995** với cùng cấu hình huấn luyện. Việc lấy vector toàn chuỗi ảnh hưởng đường gradient và representation học được.

## 3. Context và chi phí

| Length | Tokens retained | Attention memory (MiB) | Training time (s) | F1 |
| --- | --- | --- | --- | --- |
| 32 | 0.1394 | 1.0000 | 29.6608 | 0.7166 |
| 64 | 0.2756 | 4.0000 | 48.2187 | 0.7567 |
| 128 | 0.5183 | 16.0000 | 103.2435 | 0.8123 |
| 256 | 0.7719 | 64.0000 | 226.7717 | 0.8195 |

T=128 → 256 làm một tensor attention của batch 64, 4 heads tăng 16 → 64 MiB. Những con số này là dung lượng tính từ shape float32, không phải peak memory. Context dài hơn giữ thêm token nhưng không tự đảm bảo F1 tốt hơn; chi phí và noise cùng tăng.

## 4. Giới hạn khi diễn giải

Một seed, subset 10,000 train và 5 epochs chưa phản ánh mọi khả năng của kiến trúc. Cả hai baseline bị truncation; Transformer học từ đầu trên IMDb, không dùng pretrained weights. Attention cao ở một từ chỉ mô tả phân phối trọng số, không chứng minh từ đó gây ra dự đoán. Sáu mẫu đúng/sai và các giả thuyết cụ thể được ghi trong `error_analysis.md` và mục 26 của notebook.

## 5. Từ Transformer tới BERT

Transformer bổ sung cấu trúc hoàn chỉnh quanh attention: vị trí, nhiều head, FFN, residual và normalization. Mô hình vẫn cần dữ liệu để học tốt và chịu giới hạn context/chi phí. LAB 08 có thể tận dụng pretraining của BERT để chuyển representation đã học từ corpus lớn sang tác vụ sentiment; cần đánh giá lại thay vì mặc định pretraining giải quyết mọi lỗi.
