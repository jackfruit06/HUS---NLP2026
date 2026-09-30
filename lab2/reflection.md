# Reflection (Mục 24)

**Câu 1. Nếu tăng n, mô hình nhận thêm thông tin gì?**

nếu tăng n thì mô hình quan sát được nhiều từ trước đó làm context dự đoán từ tiếp theo. 

**Câu 2. Tại sao tăng n lại làm sparsity tăng?**

Tăng n lại làm sparity tăng vì khi tăng n thì số lượng n-grams cũng tăng theo. Và số n-gram quan sát được trong qua trình train sẽ bị giới hạn bởi số token, nên có nhiều n-gram có count thấp.

**Câu 3. Tại sao smoothing cần thiết?**

Smoothing cần thiết để tránh cho xác xuất của một từ chưa từng xuất hiện trong data training bằng 0 và smoothing cũng giúp điều chỉnh phân phối của những n-gram có count thấp

**Câu 4. Perplexity đo điều gì?**

Perplexity là chỉ số đo mức độ dự đoán dữ liệu test của mô hình tốt đến đâu dựa trên xác suất mà nó gán cho các từ thực tế.

**Câu 5. Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không?**

Không chắc, một model có perplexity thấp chỉ thể hiện là nó có thể gán xác suất cao cho ngôn ngữ được dự đoán từ tiếp theo dựa trên bối cảnh đã được train có độ chính xác cao chứ không cho biết là có tốt đối với con người hay không.

**Câu 6. N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?**

N-gram thất bại ở việc nó có thể dùng quá ít ngữ cảnh để có thể đưa ra dự đoán cho nội dung tiếp theo. Ngoài ra n-gram còn gặp vấn đề ở việc càng nhiều context thì độ thưa của ma trận cũng càng lớn và nó không thể xử lý tốt quan hệ giữa những từ các xa nhau.

**Câu 7 Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?**
Không bởi vì trigram chỉ cho phép sử dụng 2 từ context trước đó. Đây là giới hạn do cách mô hình n-gram được định nghĩa.

<!-- Các câu tiếp theo nằm ở trang sau trang 13 của đề — chép vào đây. -->
