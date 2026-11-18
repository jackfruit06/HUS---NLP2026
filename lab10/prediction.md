# LAB 10 — Prediction

#### 9. Prediction Before Experiment

| Prediction | Reason | Expected evidence |
| --- | --- | --- |
| 1. Nhiều keyword giống question nhưng không chứa đáp án có thể làm model nhầm span. | Lexical overlap không xác định quan hệ trả lời; heuristic dễ chọn câu distractor, Transformer cần học tương tác question/context. | EM/F1 theo nhóm lexical overlap và cặp context gốc/thêm câu lặp question. |
| 2. Answer dài thường khó đạt EM hơn. | Phải tìm đúng cả start và end; vẫn có thể nhận F1 một phần nếu bỏ sót vài từ. | EM/F1 các nhóm 1,2–3,4–6,7+ từ. |
| 3. Context thêm thông tin không liên quan có thể giảm performance. | Nhiều span cạnh tranh và số window tăng; answer có thể ra ngoài một window dù còn trong toàn context. | So sánh cùng câu hỏi và gold trên original/distractor; giữ model cố định. |
| 4. Đổi thứ tự câu có thể đổi dự đoán dù answer text còn nguyên. | Absolute position và quan hệ context thay đổi. Giữ answer text chưa chắc giữ nguyên mọi tham chiếu ngữ nghĩa. | Thử đảo câu trên cùng 50 ví dụ còn đầy đủ gold text, so EM/F1 và ghi giới hạn phép biến đổi. |
