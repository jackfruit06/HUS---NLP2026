# LAB 08 — Reflection

Sau LAB 07, tôi biết ghép attention, positional information, FFN, residual và LayerNorm thành Transformer. Tuy nhiên, xây được kiến trúc chưa có nghĩa đã có representation hữu ích. Train từ đầu buộc mô hình học cả đặc trưng ngôn ngữ và ranh giới sentiment bằng dữ liệu downstream, trong khi nhãn thường ít và tốn công thu thập.

Pretraining tạo điểm khởi đầu đã học từ corpus lớn. Trong LAB 08, BERT-Tiny fine-tuned đạt F1 0.8162, còn cùng kiến trúc khởi tạo random đạt 0.8234 trên IMDb với 10.000 mẫu train. Ở 10% dữ liệu, F1 lần lượt là 0.6776 và 0.7082; pretrained có accuracy cao hơn dù F1 thấp hơn. So sánh cả bốn mức dữ liệu cho thấy lợi ích phụ thuộc metric và chưa hoàn toàn khớp dự đoán ban đầu.

Downstream task ở đây là sentiment classification. Head mới học ánh xạ từ contextual [CLS] sang positive hoặc negative. Frozen chỉ cập nhật head; fine-tuning cập nhật cả encoder để representation phù hợp hơn với tác vụ. Chênh lệch F1 frozen 0.6793 và fine-tuned 0.8162 cho thấy cần cân nhắc mức thích nghi cùng ngân sách, thay vì mặc định một chế độ luôn tốt nhất.

Về computational cost, tái sử dụng checkpoint tránh lặp lại pretraining đắt tiền; nó không làm chi phí đó biến mất. Thời gian báo cáo chỉ đo downstream training. Tăng context cũng tốn tài nguyên: từ 128 lên 256 token, số phần tử attention mỗi head tăng bốn lần. Lợi ích performance cần được so với thời gian, bộ nhớ và yêu cầu thực tế trước khi tiếp tục tăng độ dài.

Pretrained representation vẫn có limitation. IMDb và Amazon khác domain, độ dài và cách chọn câu; F1 giảm từ 0.8162 trên IMDb xuống 0.7929 trên Amazon, nhưng khác biệt này còn đi cùng cách tuyển chọn và gán nhãn. Lỗi rất tự tin cho thấy softmax cao không bảo đảm đúng. Truncation, phủ định, sentiment lẫn lộn và tokenizer có thể ảnh hưởng dự đoán. Fine-tuning còn có nguy cơ overfit hoặc catastrophic forgetting, nhưng cần phép đo riêng để xác nhận. Vì vậy, pretrained model thường là khởi đầu tiết kiệm dữ liệu và công sức, còn đánh giá đúng tác vụ, domain và giới hạn tài nguyên vẫn là trách nhiệm không thể bỏ qua.
