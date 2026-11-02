# LAB 07 — Prediction

#### 8. Prediction trước experiment

Các dự đoán được ghi trước khi chạy thí nghiệm.

| # | Prediction | Reason | Confidence |
|---|---|---|---|
| 1 | Transformer xử lý các vị trí song song hơn RNN. | Q/K/V tính cùng lúc; RNN phụ thuộc state của timestep trước. Không đồng nghĩa luôn ít giây hơn. | Cao — 95% |
| 2 | Bỏ mọi thông tin vị trí, self-attention không biết thứ tự tuyệt đối. | Cùng trọng số áp dụng cho mọi vị trí; không có recurrence hay vị trí trong đầu vào. | Cao — 95% |
| 3 | Không PE, self-attention + mean pooling cho cùng vector với hai chuỗi là hoán vị của nhau. | Output token hoán vị tương ứng; lấy trung bình loại bỏ hoán vị. Giả sử eval, không causal mask, không yếu tố vị trí khác. | Cao — 95% |
| 4 | Hai head không nhất thiết học cùng pattern. | Các phép chiếu khác nhau; vẫn có thể trùng hoặc dư thừa. | Cao — 90% |
| 5 | Tăng số head không đảm bảo accuracy tăng. | Giữ d_model cố định làm d_k giảm; kết quả còn phụ thuộc dữ liệu và tối ưu. | Cao — 90% |
| 6 | Transformer không luôn nhanh hơn RNN ở mọi độ dài. | Attention có chi phí T², còn thời gian thực phụ thuộc phần cứng, batch và cách cài đặt. | Cao — 90% |
