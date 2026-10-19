# LAB 05 — Error Analysis (Mục 24)
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Mô hình phân tích:** LSTM Classifier (`embed_dim=64`, `hidden_size=64`, `max_length=100`)  

---

## 1. Tổng quan phương pháp đánh giá

Mục tiêu của Error Analysis không chỉ dừng lại ở việc đếm số lượng dự đoán đúng/sai, mà là **mổ xẻ cơ chế hoạt động của mô hình chuỗi (sequence model)** để trả lời câu hỏi cốt lõi:
> *Lỗi sai này xuất phát từ bản thân cơ chế sequence modeling (như vanishing gradient, thông tin bị nén, mất trật tự ngữ cảnh), hay do các yếu tố bên ngoài (như truncation cắt mất câu kết, từ hiếm OOV, châm biếm sarcasm, hoặc nhiễu nhãn trong tập dữ liệu)?*

Dưới đây là 3 mẫu dự đoán đúng tiêu biểu (Correct) và 4 mẫu dự đoán sai tiêu biểu (Incorrect) được trích xuất trực tiếp từ tập kiểm thử `test_set` của mô hình LSTM.

---

## 2. Bảng tổng hợp các mẫu phân tích

| Mã mẫu | Nhãn thực tế (True) | Dự đoán (Pred) | $P(\text{positive})$ | Phân loại nguyên nhân chính | Tác động của Sequence Modeling |
|---|:---:|:---:|:---:|---|---|
| **C1** | Negative | Negative | 0.1449 | Phủ định trực tiếp, giọng văn hoài nghi | ✅ Sequence model giữ được trạng thái tiêu cực |
| **C2** | Negative | Negative | 0.2048 | Tranh luận phản bác, cấu trúc tu từ | ✅ Nhận biết được ngữ cảnh phản biện |
| **C3** | Negative | Negative | 0.2653 | Từ khóa tiêu cực mạnh (*over-rated, over-hyped*) | ✅ Cấu trúc so sánh phủ định (*no Indiana Jones*) |
| **E1** | **Positive** | **Negative** | **0.1204** | Truncation + Cấu trúc phủ định tu từ (*nothing to do, hurts*) | ⚠️ Truncation và sự xuất hiện dày đặc của từ tiêu cực bề mặt |
| **E2** | **Positive** | **Negative** | **0.1558** | Từ ngữ mô tả cốt truyện tiêu cực (*hideous, plastic surgery*) | ⚠️ Nén thông tin cốt truyện lấn át thái độ khen ngợi |
| **E3** | **Negative** | **Positive** | **0.5970** | Phim/Game mở đầu hào hứng (*excited, first ones*), thất vọng ở cuối | ❌ **Lỗi Truncation nặng**: Đoạn chê bai ở đuôi bị cắt đứt |
| **E4** | **Negative** | **Positive** | **0.6441** | Từ vựng phức tạp, phê bình mang tính học thuật | ❌ Giới hạn từ vựng (Vocabulary) và cấu trúc câu phức |

---

## 3. Phân tích chi tiết từng trường hợp

### 3.1. Các mẫu dự đoán đúng (Correctly Classified)

#### Mẫu C1 (True: Negative, Pred: Negative, $P = 0.1449$)
- **Trích đoạn Text:**  
  > *"To truly appreciate this film you had to be there (acting?) or have been a crew member. Yes, I am 'Selena', and at the ripe old age of 42, have serious doubts about what we were doing/did. It all started out to be like a 'John Waters' type thing, fri..."*
- **Phân tích:**  
  Review thể hiện sự hoài nghi sâu sắc (*"serious doubts"*, dấu chấm hỏi mang tính châm biếm *"acting?"*). Trạng thái ẩn của LSTM tích lũy dần các biểu hiện thất vọng từ đầu câu và duy trì giá trị âm ổn định về cuối, giúp mô hình tự tin gán nhãn Negative với xác suất Positive chỉ đạt $14.49\%$.

#### Mẫu C2 (True: Negative, Pred: Negative, $P = 0.2048$)
- **Trích đoạn Text:**  
  > *"Why is it that every time I mention this movie to somebody, I have to hear 10 minutes of praise about it. I have come to the conclusion that any movie that has a twist upon a twist is deemed 'genius' by anyone who watches it. Is that really what film..."*
- **Phân tích:**  
  Mặc dù câu chứa nhiều từ tích cực có tần suất cao (*"praise"*, *"genius"*), câu văn lại được đặt trong một cấu trúc câu hỏi phản biện mang tính mỉa mai (*"Why is it that...", "Is that really what film..."*). LSTM đã không bị bẫy bởi các từ tích cực đơn lẻ nhờ cơ chế đọc tuần tự và cập nhật ngữ cảnh.

#### Mẫu C3 (True: Negative, Pred: Negative, $P = 0.2653$)
- **Trích đoạn Text:**  
  > *"National Treasure is about as over-rated and over-hyped as they come. Nicholas Cage is in no way a believable action hero, and this film is no 'Indiana Jones'..."*
- **Phân tích:**  
  Mẫu này chứa cấu trúc phủ định rõ ràng: *"in no way a believable action hero"*, *"this film is no Indiana Jones"*. LSTM xử lý tốt phạm vi tác động của từ *"no"* đứng trước danh từ tích cực (*"believable"*, *"Indiana Jones"*), đảo ngược chiều vector biểu diễn và đưa ra kết quả chính xác.

---

### 3.2. Các mẫu dự đoán sai (Incorrectly Classified)

#### Mẫu E1 (True: Positive, Pred: Negative, $P = 0.1204$) — False Negative
- **Trích đoạn Text:**  
  > *"Ignore the comment before mine. The show is comic genius, most underrated show ever. 3 reasons why. Reason 1, the flashbacks are hilarious, who cares if they have absolutely nothing to do with the story, they make me laugh so hard it hurts and isn't..."*
- **Nguyên nhân phân tích:**
  1. **Từ tiêu cực bề mặt (Surface Negative Words):** Người viết dùng cụm *"laugh so hard it hurts"* (cười đau cả ruột) và *"have absolutely nothing to do with"*. Từ *"hurts"* và *"nothing"* thường mang sắc thái tiêu cực mạnh trong tập huấn luyện.
  2. **Cấu trúc tu từ phức tạp:** Câu mở đầu *"Ignore the comment before mine"* tạo ra một ngữ cảnh phản biện kỳ lạ mà mô hình khó nắm bắt nếu không có cơ chế chú ý toàn cục.
  3. **Kết luận:** Lỗi này **không hẳn do sequence modeling**, mà do sự bất tương xứng giữa nghĩa đen từ vựng (literal meaning) và biểu cảm phóng đại mang tính khẩu ngữ (hyperbole).

#### Mẫu E2 (True: Positive, Pred: Negative, $P = 0.1558$) — False Negative
- **Trích đoạn Text:**  
  > *"Mighty Like A Moose is one of many short films Director Leo McCarey did starring Charley Chase. What a dandy it is! Charlie and his wife both undergo plastic surgery to improve their hideous appearances unbeknown-est to each other..."*
- **Nguyên nhân phân tích:**
  1. **Nội dung tóm tắt cốt truyện lấn át đánh giá:** Người viết dành phần lớn văn bản để tóm tắt một bộ phim hài kỳ quặc với các từ như *"plastic surgery"*, *"hideous appearances"*.
  2. **Từ khen ngợi cổ điển / hiếm:** Lời khen duy nhất ở đầu là *"What a dandy it is!"*. Từ *"dandy"* là một từ hiếm trong tập IMDb hiện đại, có thể bị gán vector embedding chưa chuẩn xác hoặc tần suất quá thấp.
  3. **Kết luận:** Do **hạn chế từ vựng (Vocabulary/Embedding)** và việc mô hình bị nhầm lẫn giữa lời kể nội dung bi hài kịch với thái độ khen ngợi của người xem.

#### Mẫu E3 (True: Negative, Pred: Positive, $P = 0.5970$) — False Positive
- **Trích đoạn Text:**  
  > *"When I first heared that there was going to be The World is not Enough video game for the Nintendo 64, I was so excited. When it finally came out, I was one of the first ones to rent it. I rented it for 7 days, and I got through the whole game!!! The..."*
- **Nguyên nhân phân tích:**
  1. **Rủi ro Truncation (Cắt ngắn chuỗi):** Đây là ví dụ điển hình nhất của hiện tượng **Truncation Artifact**. Tác giả mở đầu bằng sự hào hứng cao độ (*"so excited"*, *"first ones to rent it"*, *"got through the whole game!!!"*). Đoạn chê bai gay gắt của bài review nằm hoàn toàn ở nửa sau (sau token thứ 100).
  2. Vì `max_length = 100`, mô hình LSTM chỉ được nhìn thấy phần đầu đầy tích cực và hoàn toàn bị tước đoạt phần kết luận thất vọng.
  3. **Kết luận:** Lỗi này **100% xuất phát từ siêu tham số `max_length`** trong pipeline tiền xử lý, không phải do khả năng suy luận của mô hình.

#### Mẫu E4 (True: Negative, Pred: Positive, $P = 0.6441$) — False Positive
- **Trích đoạn Text:**  
  > *"The basic genre is a thriller intercut with an uncomfortable menage-a-trois. Fellowes has tried to make a lot more out of this, using the lies of the title in order to bring about all manner of small twists, invariably designed to surprise the charac..."*
- **Nguyên nhân phân tích:**
  1. **Phê bình mang tính học thuật (Literary/Subtle Criticism):** Review không dùng các từ chửi bới thông thường mà dùng lối phê bình điện ảnh gián tiếp (*"has tried to make a lot more out of this"*, *"invariably designed to surprise"* — ngụ ý phim cố làm ra vẻ nguy hiểm nhưng gượng gạo).
  2. Mô hình Recurrent ở mức phân loại nhị phân thô sơ không thể hiểu được cấu trúc câu phức hàm ẩn (subtle critique) nếu không được tiền huấn luyện trên các corpus ngôn ngữ lớn.
  3. **Kết luận:** Do **hạn chế về năng lực biểu diễn ngữ nghĩa bậc cao** của mạng nơ-ron hồi quy huấn luyện từ đầu (training from scratch).

---

## 4. Tổng kết bài học rút ra

1. **Không quy chụp mọi lỗi cho mô hình:** Cần phân định rõ lỗi do **tiền xử lý (truncation)**, lỗi do **từ vựng (OOV, rare slang)**, và lỗi do **cấu trúc ngữ nghĩa (sarcasm, subtle shift)**.
2. **Nút thắt cổ chai của `max_length`:** Với các bài review có cấu trúc *"đầu khen/hào hứng, cuối chê bai"* hoặc ngược lại, việc cắt chuỗi cố định 100 tokens sẽ loại bỏ thông tin quyết định.
3. **Sự cần thiết của cơ chế Attention:** Mô hình LSTM chỉ truyền trạng thái ẩn cuối cùng $h_T$ vào classifier khiến các chi tiết quan trọng ở giữa câu dễ bị lu mờ. Cơ chế Self-Attention sẽ cho phép bộ phân loại trực tiếp chú ý vào từ khóa then chốt bất kể vị trí của nó.
