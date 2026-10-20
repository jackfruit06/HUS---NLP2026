## Prediction 1 — Word Order Handling: TF-IDF + LR vs RNN + Embedding

- **Question:** Giữa `TF-IDF + Logistic Regression` và `RNN + Embedding`, model nào có khả năng xử lý word order trực tiếp hơn?
- **Prediction:** `RNN + Embedding` có khả năng xử lý word order trực tiếp hơn.
- **Reason:** 
  - `TF-IDF` hoạt động dựa trên giả định Bag-of-Words (BoW), tức là đếm tần suất xuất hiện của từ mà hoàn toàn bỏ qua vị trí của từ trong câu. Hai câu hoán đổi vị trí từ sẽ có vector TF-IDF unigram giống hệt nhau.
  - Ngược lại, `RNN` đọc từng token tuần tự theo thứ tự thời gian $t=1, \dots, T$. Trạng thái ẩn $h_t$ tại mỗi bước là kết quả của $h_{t-1}$ và $x_t$, do đó trạng thái cuối cùng phụ thuộc trực tiếp vào trật tự xuất hiện của các từ.
- **Confidence:** 95% 


## Prediction 2 — Long-range Dependency: RNN vs LSTM

- **Question:** Giữa `Simple RNN` và `LSTM`, model nào có khả năng xử lý dependency dài tốt hơn?
- **Prediction:** `LSTM` xử lý dependency dài tốt hơn vượt trội so với Simple RNN.
- **Reason:**
  - `Simple RNN` chịu ảnh hưởng nặng nề của hiện tượng vanishing gradient khi lan truyền ngược qua thời gian (BPTT). Do gradient bị nhân liên tiếp qua nhiều ma trận chuyển trạng thái kèm hàm phi tuyến bão hòa $\tanh$, tín hiệu lỗi suy giảm về 0 sau khoảng 10-15 bước, khiến RNN mất khả năng học quan hệ giữa các từ cách xa nhau.
  - `LSTM` giới thiệu Cell State ($c_t$) với luồng thông tin tuyến tính được kiểm soát bởi các cổng (Forget gate, Input gate). Cấu trúc cộng $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ tạo ra một đường cao tốc dẫn gradient (gradient highway), giúp gradient lan truyền ngược qua hàng chục bước thời gian mà không bị triệt tiêu khi $f_t \approx 1$.
- **Confidence:** 98%



## Prediction 3 — Ảnh hưởng của Sequence Length (50 → 200 tokens)

- **Question:** Nếu sequence length tăng từ $50 \to 200$, quá trình training của RNN sẽ thay đổi như thế nào?
- **Prediction:** 
  1. Thời gian huấn luyện (Training time) sẽ tăng tỉ lệ thuận xấp xỉ tuyến tính (tăng khoảng 3-4 lần).
  2. Lượng bộ nhớ GPU/RAM sử dụng cho activation cache (để phục vụ BPTT) tăng gấp 4 lần.
  3. Nguy cơ vanishing gradient của Simple RNN sẽ tăng mạnh; trong khi LSTM có thể tận dụng thêm ngữ cảnh bổ sung ở các câu dài nhưng cũng đối mặt với nhiều noise và padding hơn.
- **Reason:** Độ phức tạp tính toán của thuật toán BPTT trong RNN là $O(T \times B \times d^2)$, với $T$ là độ dài chuỗi, $B$ là batch size, $d$ là hidden size. Khi $T$ tăng từ 50 lên 200 ($4\times$), số phép nhân ma trận tăng gấp 4 lần.
- **Confidence:** 90%


## Prediction 4 — Tác động của việc Truncate review về 30 tokens

- **Question:** Nếu truncate toàn bộ review về 30 tokens, thông tin classification có thể bị ảnh hưởng không?
- **Prediction:** Có, hiệu năng phân loại (Accuracy và F1) sẽ bị **giảm sút nghiêm trọng**.
- **Reason:**
  - Theo thống kê từ IMDb dataset, độ dài trung vị (median) của một bài review là khoảng 230 tokens, và giá trị trung bình là 310 tokens. Truncate về 30 tokens đồng nghĩa với việc vứt bỏ hơn 85-90% nội dung bài viết.
  - Nhiều bài đánh giá phim mở đầu bằng việc tóm tắt cốt truyện, diễn biến dài dòng và chỉ đưa ra kết luận đánh giá (khen/chê, kết luận cuối) ở những câu cuối cùng. Cắt cụt chuỗi ở token thứ 30 sẽ làm mất hoàn toàn kết luận và các từ khóa chuyển ý nằm ở phần sau.
- **Confidence:** 95%

## Prediction 5 — LSTM có chắc chắn cho Accuracy cao hơn RNN trên mọi dataset không?

- **Question:** LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?
- **Prediction:** KHÔNG chắc chắn.
- **Reason:**
  - Mặc dù LSTM có ưu thế vượt trội về mặt lý thuyết trong việc giải quyết long-range dependencies, nhưng LSTM có số lượng tham số gấp 4 lần Simple RNN ($4 \times (d^2 + d \cdot x)$ so với $d^2 + d \cdot x$).
  - Trên các dataset có dung lượng rất nhỏ (vài trăm mẫu) hoặc văn bản cực ngắn (như tweet dưới 15 từ), cấu trúc nhiều tham số của LSTM dễ dẫn đến overfitting nghiêm trọng. Trong các tình huống này, Simple RNN (với cấu trúc đơn giản, ít tham số hơn) có thể cho kết quả tương đương hoặc thậm chí khái quát hóa tốt hơn.
  - Ngoài ra, việc tinh chỉnh hyperparameter (learning rate, weight decay, dropout) ảnh hưởng rất lớn đến kết quả thực nghiệm.
- **Confidence:** 90%
