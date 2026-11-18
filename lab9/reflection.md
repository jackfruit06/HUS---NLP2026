# LAB 09 — Reflection

Sau LAB 08, tôi dùng pretrained Transformer để dự đoán một nhãn sentiment cho cả review. Sang LAB 09, encoder vẫn tạo contextual representation tại mọi vị trí. Điểm thay đổi nằm ở cách đọc các vector đó và mục tiêu học của prediction head. Vì vậy, cùng một kiến trúc encoder có thể phục vụ nhiều bài toán mà không cần thiết kế lại toàn bộ mạng.

Với sequence classification, bài trước lấy vector ở vị trí [CLS], đưa qua Linear rồi dự đoán positive hoặc negative. Trong token classification, Linear được áp dụng chung cho từng vector trong chuỗi. Nếu hidden states có shape B×T×d thì logits có shape B×T×C. Mỗi vị trí nhận chín điểm số BIO thay vì cả câu chỉ nhận hai điểm số sentiment. Đổi head cũng đồng nghĩa đổi nhãn đích, loss và cách đánh giá; không phải một head mới tự hoạt động tốt ngay khi khởi tạo.

Pretraining cung cấp điểm xuất phát đã học từ văn bản, còn fine-tuning điều chỉnh representation cho ranh giới và loại entity. Trong thí nghiệm, baseline entity F1 đạt 0.6912, BERT-Tiny đạt 0.7851. Các số đo này phản ánh hai cấu hình trên cùng CoNLL-2003 test, không chứng minh encoder hiểu mọi tên riêng. Baseline có đặc trưng chữ hoa và từ lân cận, trong khi tokenizer BERT-Tiny uncased bỏ thông tin hoa thường, nên sự so sánh còn gắn với lựa chọn biểu diễn.

Khó khăn mới là annotation theo word nhưng encoder đọc subword. Tôi chọn chỉ giám sát mảnh đầu, đặt -100 cho phần còn lại và special tokens. Những mảnh bị bỏ khỏi loss vẫn tham gia attention. Khi đánh giá, dự đoán được ghép về word; entity chỉ đúng nếu khớp cả loại lẫn hai đầu mút. Token accuracy cao do nhiều nhãn O không thay thế được entity F1.

Pipeline IE chuyển span sang PERSON, ORG và LOCATION. Gold của ba mươi câu tự biên soạn giúp đo lỗi cụ thể, nhưng câu theo mẫu và tên nổi tiếng còn đơn giản hơn tin tức tự nhiên. Entity hiếm, tên mơ hồ, domain mới và context vượt chunk vẫn là giới hạn. Tái sử dụng encoder tiết kiệm công sức xây kiến trúc, còn dữ liệu gán nhãn, alignment và tiêu chí đánh giá quyết định liệu mô hình có hữu ích cho tác vụ thực tế hay không.
