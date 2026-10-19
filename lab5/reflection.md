# LAB 05 — Reflection & Synthesis
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Chủ đề:** Từ Static Representations đến Sequential Modeling  

---

## 1. Đối chiếu Dự đoán ban đầu (Predictions) với Kết quả thực nghiệm

| Dự đoán ban đầu (Mục 10) | Dự đoán | Kết quả thực nghiệm thực tế | Nhận xét & Đánh giá |
|---|:---:|:---:|---|
| **Prediction 1:** Khả năng xử lý Word Order (TF-IDF vs RNN) | RNN xử lý trật tự từ trực tiếp hơn | **Khớp về lý thuyết, nhưng bất ngờ về hiệu năng thực tế.** Cặp câu hoán đổi cho thấy unigram TF-IDF bất biến, trong khi RNN có độ nhạy thứ tự. Tuy nhiên F1 của TF-IDF (~0.88) cao hơn RNN (~0.47). | Kiến trúc có khả năng xử lý thứ tự không đồng nghĩa với việc mô hình học được phân loại tốt hơn nếu bị hạn chế bởi vanishing gradient. |
| **Prediction 2:** Khả năng xử lý Long Dependency (RNN vs LSTM) | LSTM vượt trội hơn Simple RNN | **Chính xác.** LSTM đạt F1 = 0.5818, GRU đạt F1 = 0.7420, trong khi Simple RNN chỉ đạt F1 = 0.4717. | Cấu trúc Gate và Cell State của LSTM/GRU đóng vai trò then chốt trong việc duy trì dòng gradient qua thời gian. |
| **Prediction 3:** Tác động của Sequence Length ($50 \to 200$) | Thời gian tăng tuyến tính, bộ nhớ tăng | **Chính xác.** Thời gian train tăng từ 12.43s (L=50) lên 25.10s (L=100) và 44.82s (L=200). Tỷ lệ token giữ lại tăng từ 21.7% lên 66.8%. | Thời gian tính toán tỉ lệ thuận với độ dài chuỗi $T$. Tuy nhiên F1 không tăng tuyến tính do nhiều noise/padding hơn. |
| **Prediction 4:** Cắt cụt review về 30 tokens | Hiệu năng giảm sút nghiêm trọng | **Chính xác.** Truncation ngắn làm mất hoàn toàn kết luận bài review, khiến mô hình bị nhầm lẫn nghiêm trọng như đã chỉ ra trong Error Analysis. | Việc chọn `max_length` phải dựa trên thống kê phân phối độ dài (median ~230) chứ không được chọn tùy tiện. |
| **Prediction 5:** LSTM có chắc chắn thắng RNN trên mọi dataset? | Không chắc chắn trên mọi trường hợp | **Chính xác.** LSTM cần nhiều dữ liệu và thời gian huấn luyện hơn để hội tụ tốt các trọng số cổng so với cấu trúc đơn giản. | Mô hình phức tạp nhiều tham số hơn cần lượng dữ liệu tương ứng để tránh overfitting. |

---

## 2. Những kết quả bất ngờ (Surprising Findings)

### 2.1. Tại sao TF-IDF + Logistic Regression lại vượt trội hơn Simple RNN?
- **Thực tế:** TF-IDF + LR đạt **Accuracy 88.15%**, trong khi Simple RNN chỉ đạt **50.95%** (gần tương đương đoán ngẫu nhiên trên bài toán nhị phân).
- **Lý do chuyên sâu:**
  1. **Tín hiệu cảm xúc cục bộ (Local Sentiment Clues):** Trong bài toán phân loại sắc thái phim, phần lớn thông tin nằm ở các từ vựng mang cực tính mạnh (*"masterpiece"*, *"terrible"*, *"boring"*). TF-IDF gom toàn bộ các từ này trong toàn bộ văn bản (lên tới 10,000 features) mà không bị giới hạn độ dài.
  2. **Vanishing Gradient trong Simple RNN:** Simple RNN bị suy giảm gradient sau 10-15 timestep. Do đó, với `max_length = 100`, mô hình hầu như "mù" trước các từ xuất hiện ở nửa đầu văn bản và không thể tối ưu hóa trọng số hiệu quả.

### 2.2. Tại sao GRU lại đạt kết quả cao hơn LSTM trong lần chạy này?
- GRU đạt F1 **0.7420**, vượt qua LSTM (**0.5818**).
- Với tập dữ liệu mẫu và số epoch nhỏ (5 epochs), GRU có ít tham số hơn ($3 \times d^2$ so với $4 \times d^2$ của LSTM), giúp GRU hội tụ nhanh hơn, ít bị phân mảnh gradient và thích ứng tốt hơn với cấu hình siêu tham số hiện tại.

---

## 3. Bản chất của Sequence Modeling: Context vs Computational Cost

Huấn luyện mô hình chuỗi luôn đòi hỏi sự đánh đổi giữa hai yếu tố:
1. **Context (Bối cảnh ngữ nghĩa):** Chuỗi càng dài, mô hình càng giữ được nhiều từ vựng, bao quát được cả phần mở đầu, diễn biến và kết luận của bài viết.
2. **Computational Cost (Chi phí tính toán & Bộ nhớ):**
   - Độ phức tạp tính toán: $O(T \times B \times d^2)$ cho mỗi epoch.
   - Bộ nhớ lưu trữ activations phục vụ BPTT tăng tỉ lệ thuận với $T$.
   - Khi $T$ tăng từ 50 lên 200, thời gian train tăng gấp gần 4 lần nhưng F1 không tăng tương ứng do hiện tượng nén thông tin (information bottleneck).

---

## 4. Động lực chuyển dịch: Từ RNN/LSTM sang Attention Mechanism và Transformer

Mặc dù LSTM và GRU đã cải thiện đáng kể bài toán vanishing gradient so với Simple RNN, kiến trúc Recurrent vẫn tồn tại **hai nút thắt cổ chai không thể vượt qua**:

```
[Token 1] ──> [Token 2] ──> [Token 3] ──> ... ──> [Token T]
                                                        │
                                                        ▼
                                                  Vector h_T duy nhất (Information Bottleneck)
                                                        │
                                                        ▼
                                                   Classifier
```

### 1. Nút thắt cổ chai thông tin (Information Bottleneck)
- Trong mô hình Recurrent, toàn bộ ý nghĩa của một văn bản gồm hàng trăm từ buộc phải nén tuần tự vào một vector trạng thái ẩn duy nhất $h_T$ tại bước cuối cùng.
- Khi chuỗi dài, các thông tin chi tiết ở đầu chuỗi khó tránh khỏi việc bị ghi đè hoặc phai nhạt dần.

### 2. Nút thắt cổ chai tính toán tuần tự (Sequential Bottleneck)
- Tại bước $t$, mô hình **bắt buộc phải đợi** trạng thái ẩn $h_{t-1}$ của bước trước đó hoàn thành mới có thể tính tiếp $h_t$.
- Điều này ngăn cản hoàn toàn việc song song hóa (parallelization) trên GPU/TPU, khiến việc mở rộng quy mô dữ liệu và mô hình trở nên vô cùng tốn kém về thời gian.

### Giải pháp đột phá: Attention Mechanism & Transformer
- **Cơ chế Attention (Bahdanau et al., 2014; Vaswani et al., 2017)** cho phép mô hình kết nối trực tiếp đầu ra với *mọi* trạng thái ẩn $h_1, \dots, h_T$ thông qua trọng số chú ý động.
- Kiến trúc **Transformer** loại bỏ hoàn toàn liên kết đệ quy tuần tự, thay thế bằng **Multi-Head Self-Attention**, cho phép xử lý toàn bộ các từ trong văn bản cùng một lúc (song song hoàn toàn trên GPU) và kết nối trực tiếp bất kỳ hai từ nào trong câu mà không bị cản trở bởi khoảng cách thời gian.
