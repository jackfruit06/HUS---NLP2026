# LAB 06 — Prediction trước Experiment
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026

Các dự đoán dưới đây được ghi trước khi chạy notebook. Kết quả và đối chiếu được trình bày riêng trong `reflection.md`.

## Prediction 1 — Key có dot product lớn hơn

- **Question:** Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ như thế nào?
- **Prediction:** Key đó nhận weight lớn nhất trong cùng hàng attention, nếu không bị mask và temperature dương.
- **Reason:** Hàm mũ tăng đơn điệu; các phần tử chia cùng một mẫu số dương. Tuy nhiên, “lớn nhất” không nhất thiết có nghĩa là chiếm gần toàn bộ trọng số.
- **Confidence:** 100% trong các điều kiện trên.

## Prediction 2 — Tất cả scores giống nhau

- **Question:** Với $s=[1,1,1,1]$, attention weights bằng bao nhiêu?
- **Prediction:** $\alpha=[0.25,0.25,0.25,0.25]$.
- **Reason:** Bốn số hạng $e^1$ bằng nhau nên mỗi key nhận $1/4$ tổng trọng số. Entropy đạt $\ln 4$.
- **Confidence:** 100% khi không có mask.

## Prediction 3 — Một score vượt trội

- **Question:** Với $s=[1,1,10]$, attention tập trung như thế nào?
- **Prediction:** Token thứ ba nhận hơn 99.9% trọng số khi temperature bằng 1.
- **Reason:** Tỷ số trọng số giữa token thứ ba và mỗi token còn lại bằng $e^{10-1}=e^9$. Softmax gần one-hot nhưng các weight còn lại vẫn dương về mặt toán học.
- **Confidence:** 100% với score vector đã cho.

## Prediction 4 — Sequence tăng từ 10 lên 1000 tokens

- **Question:** Chi phí tính attention thay đổi như thế nào?
- **Prediction:** Full self-attention có số phần tử ma trận tăng **10,000 lần**, từ $10^2=100$ lên $1000^2=1,000,000$. Với một query cố định, số score chỉ tăng 100 lần.
- **Reason:** Self-attention ghép mọi query với mọi key; phần tính $QK^T$ và $AV$ có độ phức tạp $O(T^2d)$ khi giữ dimension cố định. Runtime thực tế không nhất thiết tăng đúng 10,000 lần do overhead, bộ nhớ và phép nhân ma trận.
- **Confidence:** 100% về số phần tử; 90% về xu hướng runtime tăng, không dự đoán tỷ lệ thời gian chính xác.

## Prediction 5 — Attention và từ quan trọng theo con người

- **Question:** Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
- **Prediction:** Không. Ví dụ model có thể đặt weight cao vào một từ thường gặp thay vì `not` trong câu phủ định.
- **Reason:** Trọng số phụ thuộc query, key, dữ liệu và mục tiêu huấn luyện. Giá trị value và classifier cũng ảnh hưởng prediction; heatmap không phải lời giải thích nhân quả. Cần đối chiếu xác suất và thử thay token để kiểm tra hypothesis.
- **Confidence:** 95% về việc có thể quan sát chênh lệch trong mô hình nhỏ huấn luyện từ đầu; không mặc định mọi attention pattern khác trực giác đều gây lỗi.
