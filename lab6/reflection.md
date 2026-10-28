# Reflection 
## 1. Đối chiếu Predictions với kết quả thực nghiệm

| Dự đoán ban đầu | Kết quả thực tế | Nhận xét & Đánh giá |
|---|---|---|
| **Prediction 1:** Key có score lớn nhất nhận weight lớn nhất | Với scores [1,2,3], weights = [0.090031,0.244728,0.665241] | Phù hợp. Weight lớn nhất là 0.665241, chưa phải toàn bộ trọng số. Kết luận trong cùng một hàng, không mask key đó, temperature dương. |
| **Prediction 2:** Scores [1,1,1,1] cho phân phối đều | Weights = [0.25,0.25,0.25,0.25] | Phù hợp. Query không ưu tiên key nào theo score đã cho; entropy = ln(4). |
| **Prediction 3:** Scores [1,1,10] làm attention tập trung gần hết ở token thứ ba | Weights ≈ [0.00012338,0.00012338,0.99975324] | Phù hợp. Token thứ ba nhận khoảng 99.9753%, không bằng 100% về mặt toán học. |
| **Prediction 4:** Tăng 10 → 1000 token làm ma trận self-attention tăng 10,000 lần | Số phần tử tăng 100 → 1,000,000; median forward CPU tăng khoảng 0.0317 → 2.8869 ms | Đúng về số phần tử. Runtime tăng khoảng 91.07 lần trong benchmark này, không phải 10,000 lần vì overhead và hiệu suất phép nhân ma trận. Một query cố định chỉ tăng số score 100 lần. |
| **Prediction 5:** Attention không luôn trùng từ quan trọng theo con người | Mẫu E1 đặt weight cao ở `masterpiece` dù lời khen hướng tới bản phim khác; E2 nhấn `horrible` trong “not a lot of horrible…”; câu “I like this movie” nhấn `i` | Phù hợp với các ví dụ quan sát được. Heatmap chưa đủ chứng minh nguyên nhân lỗi; perturbation trong E1/E2 cung cấp thêm bằng chứng về độ nhạy của model. |

Năm dự đoán đều phù hợp trong phạm vi đã nêu. Ba câu đầu dựa trên tính chất toán học nên đây là kiểm chứng phép tính; không coi chúng là bằng chứng tổng quát về chất lượng NLP. Không sửa lạVi dự đoán ban đầu để khớp output.

## 2. Scaling và temperature thay đổi điều gì?

Với scores [4,2,0], chia căn $d_k=4$ cho [2,1,0]. Max weight giảm 0.866813 → 0.665241, entropy tăng 0.441057 → 0.832396 nat. Thứ tự ưu tiên giữ nguyên nhưng distribution ít nhọn hơn.

Với cùng scores [1,2,3], temperature 0.5/1/2 cho max weight lần lượt 0.866813 / 0.665241 / 0.506480 và entropy 0.441057 / 0.832396 / 1.020191. Temperature cao không làm model “thông minh hơn”; nó chỉ thay đổi độ tập trung của weights trong phép thử cố định này.

Khi đổi query sentiment sang topic và giữ K/V, token nhận weight lớn nhất đổi từ good sang movie. Đây là ví dụ giả lập để thấy attention phụ thuộc query, không phải một danh sách từ quan trọng cố định cho cả câu. Các tên sentiment/topic do người làm quy ước cho hai tọa độ, không phải ngữ nghĩa tự xuất hiện từ training.

## 3. Những kết quả bất ngờ

### 3.1. Attention cao có thể hỗ trợ một hypothesis nhưng không giải thích toàn bộ model

Trong E1, thay `masterpiece` ở vị trí 56 bằng UNK làm P(positive) giảm 0.8003 → 0.4337, đổi dự đoán thành Negative. Trong E2, thay `horrible` ở vị trí 61 làm P(positive) tăng 0.4376 → 0.8260, đổi thành Positive. Thay vị trí weight thấp chỉ thay đổi khoảng −0.0108 và −0.0228 tương ứng, không đổi nhãn.

Điều này phù hợp với hypothesis model nhạy với từ cảm xúc nhưng chưa xử lý tốt đối tượng lời khen/phạm vi phủ định. Tuy nhiên, thay một ID làm thay cả Q/K/V và distribution, không chỉ một weight. UNK là can thiệp input, không phải chứng minh nhân quả riêng của attention weight.

### 3.2. Attention pattern không tự là linguistic explanation

Với embedding giả lập và các ma trận chiếu đang dùng, top-3 keys là:

| Query | Key thứ nhất | Key thứ hai | Key thứ ba |
|---|---|---|---|
| cat | adopted: 0.153717 | that: 0.149673 | sleeping: 0.144463 |
| adopted | adopted: 0.223895 | that: 0.204026 | sleeping: 0.180127 |
| sleeping | adopted: 0.201719 | that: 0.188644 | sleeping: 0.172775 |

Các hàng có phân phối khác nhau vì mỗi token tạo query khác nhau; cùng một sequence không có một danh sách attention duy nhất cho mọi query.

**Heatmap có chứng minh model hiểu ngữ pháp không?** Không. Đây là kết quả của vector ngẫu nhiên và phép nhân ma trận, không phải mô hình đã học từ dữ liệu ngôn ngữ. Nếu một liên kết tình cờ hợp nghĩa câu thì cũng chưa có bằng chứng hiểu ngữ pháp. Thay seed hoặc embedding có thể đổi pattern. Attention từ classifier đã huấn luyện được xem riêng ở Application và Error Analysis.


## 4. Overfitting và cách chọn checkpoint

Cả hai model chọn **epoch 3 theo validation loss**:

- LSTM: train loss tiếp tục giảm 0.5120 → 0.3094 từ epoch 3 tới 5, nhưng validation loss tăng 0.5754 → 0.5863, có dao động mạnh ở epoch 4 (0.6351).
- Attention: train loss giảm 0.4152 → 0.2548, validation loss tăng 0.4737 → 0.4999. Validation accuracy lại cao hơn ở epoch 4, nhưng quy tắc lựa chọn đã cố định theo loss nên vẫn dùng epoch 3.

Đây là dấu hiệu khoảng cách train/validation tăng, phù hợp với khả năng overfitting và/hoặc confidence kém hiệu chỉnh hơn. Không xem test trong quá trình chọn checkpoint. Số đo chỉ một seed, chưa đủ đánh giá độ ổn định hay ý nghĩa thống kê của chênh lệch hai mô hình.

