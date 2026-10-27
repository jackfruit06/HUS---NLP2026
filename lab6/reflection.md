# LAB 06 — Reflection & Synthesis
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Chủ đề:** From Recurrent State to Context-Aware Representation

---

## 1. Đối chiếu Predictions với kết quả thực nghiệm

| Dự đoán ban đầu | Kết quả thực tế | Nhận xét & Đánh giá |
|---|---|---|
| **Prediction 1:** Key có score lớn nhất nhận weight lớn nhất | Với scores [1,2,3], weights = [0.090031,0.244728,0.665241] | **Phù hợp.** Weight lớn nhất là 0.665241, chưa phải toàn bộ trọng số. Kết luận trong cùng một hàng, không mask key đó, temperature dương. |
| **Prediction 2:** Scores [1,1,1,1] cho phân phối đều | Weights = [0.25,0.25,0.25,0.25] | **Phù hợp.** Query không ưu tiên key nào theo score đã cho; entropy = ln(4). |
| **Prediction 3:** Scores [1,1,10] làm attention tập trung gần hết ở token thứ ba | Weights ≈ [0.00012338,0.00012338,0.99975324] | **Phù hợp.** Token thứ ba nhận khoảng 99.9753%, không bằng 100% về mặt toán học. |
| **Prediction 4:** Tăng 10 → 1000 token làm ma trận self-attention tăng 10,000 lần | Số phần tử tăng 100 → 1,000,000; median forward CPU tăng khoảng 0.0294 → 2.3687 ms | **Đúng về số phần tử.** Runtime tăng khoảng 80.57 lần trong benchmark này, không phải 10,000 lần vì overhead và hiệu suất phép nhân ma trận. Một query cố định chỉ tăng số score 100 lần. |
| **Prediction 5:** Attention không luôn trùng từ quan trọng theo con người | Mẫu E1 đặt weight cao ở `masterpiece` dù lời khen hướng tới bản phim khác; E2 nhấn `horrible` trong “not a lot of horrible…”; câu “I like this movie” nhấn `i` | **Phù hợp với các ví dụ quan sát được.** Heatmap chưa đủ chứng minh nguyên nhân lỗi; perturbation trong E1/E2 cung cấp thêm bằng chứng về độ nhạy của model. |

Năm dự đoán đều phù hợp trong phạm vi đã nêu. Ba câu đầu dựa trên tính chất toán học nên đây là kiểm chứng phép tính; không coi chúng là bằng chứng tổng quát về chất lượng NLP. Không sửa lại dự đoán ban đầu để khớp output.

## 2. Scaling và temperature thay đổi điều gì?

Với scores [4,2,0], chia căn $d_k=4$ cho [2,1,0]. Max weight giảm **0.866813 → 0.665241**, entropy tăng **0.441057 → 0.832396 nat**. Thứ tự ưu tiên giữ nguyên nhưng distribution ít nhọn hơn.

Với cùng scores [1,2,3], temperature 0.5/1/2 cho max weight lần lượt **0.866813 / 0.665241 / 0.506480** và entropy **0.441057 / 0.832396 / 1.020191**. Temperature cao không làm model “thông minh hơn”; nó chỉ thay đổi độ tập trung của weights trong phép thử cố định này.

Khi đổi query sentiment sang topic và giữ K/V, token nhận weight lớn nhất đổi từ **good** sang **movie**. Đây là ví dụ giả lập để thấy attention phụ thuộc query, không phải một danh sách từ quan trọng cố định cho cả câu. Các tên sentiment/topic do người làm quy ước cho hai tọa độ, không phải ngữ nghĩa tự xuất hiện từ training.

## 3. Kết quả so sánh LSTM và Attention

| Model | Accuracy | Precision | Recall | F1 | Training time | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.7105 | 0.6741 | 0.8162 | 0.7384 | 13.56 s | 2,178,306 |
| Attention | 0.7775 | 0.8124 | 0.7223 | 0.7647 | 13.88 s | 2,157,314 |

**Attention tốt hơn về accuracy và F1 trong cấu hình này**, nhưng recall positive thấp hơn. Confusion matrix cho thấy Attention có 167 false positives, 278 false negatives; LSTM có 395 false positives, 184 false negatives. Vì vậy, câu “Attention tốt hơn mọi mặt” không đúng với kết quả.

Không thể chỉ từ bảng này kết luận attention xử lý long-range dependency tốt hơn: cả hai model chỉ đọc 100 token đầu và chưa có đánh giá phụ thuộc dài được kiểm soát riêng. Lợi ích kiến trúc là đường tương tác trực tiếp giữa token; mức độ khai thác được cần thêm thí nghiệm.

**Thời gian:** Attention 13.88 s so với LSTM 13.56 s, gần nhau ở CPU 4 threads với T=100. Tính song song về kiến trúc không đảm bảo nhanh hơn trong mọi kích thước batch/phần cứng. Không dùng thời gian này để dự đoán GPU hoặc sequence rất dài.

**Phạm vi so sánh LAB 05:** LSTM được huấn luyện lại, lấy state tại token thật cuối và chọn checkpoint theo validation loss. Do quy trình chọn checkpoint và môi trường khác, không quy chênh lệch với bảng LSTM cũ trong LAB 05 chỉ cho cơ chế attention.

## 4. Những kết quả bất ngờ

### 4.1. Accuracy IMDb tương đối tốt nhưng câu ngắn rất dễ vẫn sai

| Câu | Nhãn kỳ vọng | Prediction | P(positive) |
|---|---|---|---:|
| I like this movie. | Positive | Negative | 0.001105 |
| I do not like this movie. | Negative | Negative | 0.000957 |
| I thought this movie would be bad, but it was actually excellent. | Positive | Negative | 0.000087 |

Kết quả này đáng chú ý hơn một heatmap trông hợp lý: model cho cùng nhãn Negative với hai câu có/không có phủ định. Dự đoán đúng câu thứ hai **chưa chứng minh model hiểu “not”**, vì câu positive tương ứng cũng bị gán Negative.

Một hypothesis là model học các tín hiệu từ vựng/vị trí phù hợp với review dài hơn và tổng quát hóa kém sang câu cực ngắn. Train subset có median 172 tokens, 87.25% review bị cắt về 100 token; Application chỉ có 4, 6 và 12 token. Đây là chênh lệch phân phối có thể liên quan, chưa chứng minh được nếu không có ablation/đánh giá theo độ dài. Xác suất gần 0 vẫn có thể sai nên không đồng nhất softmax confidence với độ tin cậy đã hiệu chỉnh.

### 4.2. Attention cao có thể hỗ trợ một hypothesis nhưng không giải thích toàn bộ model

Trong E1, thay `masterpiece` ở vị trí 56 bằng UNK làm P(positive) giảm **0.8003 → 0.4337**, đổi dự đoán thành Negative. Trong E2, thay `horrible` ở vị trí 61 làm P(positive) tăng **0.4376 → 0.8260**, đổi thành Positive. Thay vị trí weight thấp chỉ thay đổi khoảng **−0.0108** và **−0.0228** tương ứng, không đổi nhãn.

Điều này phù hợp với hypothesis model nhạy với từ cảm xúc nhưng chưa xử lý tốt đối tượng lời khen/phạm vi phủ định. Tuy nhiên, thay một ID làm thay cả Q/K/V và distribution, không chỉ một weight. UNK là can thiệp input, không phải chứng minh nhân quả riêng của attention weight.

### 4.3. Attention pattern không tự là linguistic explanation

- Query `cat` chú ý `cat` (0.5008), `sleeping` (0.1826), `the` (0.0959).
- Query `adopted` chú ý `yesterday` (0.2968), `cat` (0.2201), `adopted` (0.1448).
- Query `sleeping` chú ý `cat` (0.4258), `adopted` (0.1559), `sleeping` (0.1529).

Một số liên kết có vẻ phù hợp nghĩa câu. Tuy nhiên, ở câu về book, query `book` lại tập trung rất mạnh vào `was` (khoảng 0.86), không phải `expensive`. Model được train bằng nhãn sentiment, không phải nhãn dependency; không đủ bằng chứng để nói nó hiểu ngữ pháp từ hai hình này.

## 5. Overfitting và cách chọn checkpoint

Cả hai model chọn **epoch 3 theo validation loss**:

- LSTM: train loss tiếp tục giảm 0.5120 → 0.3094 từ epoch 3 tới 5, nhưng validation loss tăng 0.5754 → 0.5863, có dao động mạnh ở epoch 4 (0.6351).
- Attention: train loss giảm 0.4152 → 0.2548, validation loss tăng 0.4737 → 0.4999. Validation accuracy lại cao hơn ở epoch 4, nhưng quy tắc lựa chọn đã cố định theo loss nên vẫn dùng epoch 3.

Đây là dấu hiệu khoảng cách train/validation tăng, phù hợp với khả năng overfitting và/hoặc confidence kém hiệu chỉnh hơn. Không xem test trong quá trình chọn checkpoint. Số đo chỉ một seed, chưa đủ đánh giá độ ổn định hay ý nghĩa thống kê của chênh lệch hai mô hình.

## 6. LAB 05 → LAB 06: giải quyết gì và phát sinh gì?

| Property | RNN/LSTM/GRU | Self-attention |
|---|---|---|
| Thứ tự token | Có thông qua recurrence | Cần positional information hoặc cơ chế bổ sung |
| Biểu diễn ngữ cảnh | Hidden state cập nhật tuần tự | Weighted combination of values theo query |
| Truyền thông tin xa | Đi qua nhiều bước; gates giúp bảo toàn thông tin | Có đường tương tác trực tiếp nếu không bị mask |
| Tính song song | Hạn chế theo timestep | Tính các query/key song song được |
| Chi phí theo T | Gần tuyến tính khi dimension cố định | Full attention bậc hai theo T |
| Nén sequence | Final hidden state | Trong bài này vẫn mean-pool context về vector cố định |

**Word order:** Khi tắt positional encoding, “dog bites man” và “man bites dog” cho P(positive) xấp xỉ **0.0566593** ở cả hai câu. Khi bật position, xác suất lần lượt **0.0004592** và **0.0002437**. Mô hình có độ nhạy thứ tự nhưng vẫn cho cùng nhãn; đây không phải bài kiểm tra hiểu vai chủ thể/đối tượng. Tắt position lúc inference cũng không phải ablation huấn luyện công bằng.

**Giới hạn dữ liệu:** Chỉ giữ 41.86% tổng số token train. Attention không thể chú ý tới phần câu đã bị cắt. Trong E3, kết luận “I gave it 10/10” nằm sau token thứ 100; việc mất kết luận là yếu tố có thể góp phần vào lỗi, không phải lời giải thích duy nhất vì đầu review đã có “blended perfectly together”.

**Giới hạn tài nguyên:** T=1000 có 1,000,000 score, T=10000 có 100,000,000 score. Một matrix float32 đã cần khoảng 381.47 MiB ở T=10000; batch, nhiều heads, gradients còn tăng bộ nhớ. Có thể song song hóa không có nghĩa attention miễn phí.

## 7. Attention → Transformer và hướng cải thiện

LAB 06 làm rõ cơ chế cốt lõi Q/K/V → scores → softmax → context. Transformer kết hợp nhiều heads, position, feed-forward blocks, residual connections và normalization để xây dựng các lớp contextual representation sâu hơn. Classifier một head trong bài không đại diện cho hiệu năng Transformer pretrained.

Nếu tiếp tục phát triển bài: đánh giá nhiều seed; chọn hyperparameters trên validation; kiểm tra riêng negation/contrast và nhóm độ dài; thử positional encoding/mean pooling bằng các model được train riêng; tăng context có kiểm soát và so sánh nhiều cách truncation. Với error analysis, có thể thêm nhiều perturbation giữ nghĩa hoặc thay từ đối chứng thay vì chỉ UNK. Các hướng này chưa được chạy trong kết quả đang báo cáo.
