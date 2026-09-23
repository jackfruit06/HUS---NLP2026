# Reflection

**1. Prediction nào của em sai?**

Prediction về sparsity sai khá xa: dự đoán ban đầu là zero-entries khoảng 85%, nhưng kết quả chạy trên corpus 30K là 99.96%. Cùng với đó , prediction về vocabulary size bị lệch gần như gấp đôi: dự đoán ~200,000 unique terms, số thật là 473,388 sau khi đã thực hiện đưa về chữ thường và tách theo khoảng trắng.

**2. Kết quả nào bất ngờ nhất?**

Bất ngờ nhất là ở Part F: Pipeline C (Extended — normalization + subword tokenization) có search performance thấp nhất (mean P@5 = 0.450) trong khi vocabulary của nó lại nhỏ nhất. Theo bản năng thì ai cũng nghĩ là"preprocessing càng kỹ, representation càng gọn và càng tốt", nhưng thực nghiệm cho thấy việc tách hậu tố của từ thành 2 từ khác nhau làm token của query và token của document dễ lệch nhau hơn, làm giảm chất lượng tìm kiếm dù index gọn hơn. 

**3. Experiment nào cung cấp evidence mạnh nhất?**

Part F (Preprocessing Ablation) cung cấp evidence mạnh nhất, vì nó so sánh trực tiếp 3 pipeline bằng cùng một bộ metric định lượng (vocabulary size, avg tokens/document, sparsity, OOV rate, search performance) trên cùng một corpus và cùng bộ eval query. 

**4. Failure case quan trọng nhất là gì?**

Query "heart attack" hoàn toàn bỏ sót các document y khoa dùng thuật ngữ chuyên môn "myocardial infarction" similarity = 0 dù nội dung liên quan trực tiếp. Nguyên nhân gốc rễ là TF-IDF/cosine similarity chỉ so khớp ở dựa trên mặt chữ query mà không có khái niệm về từ đồng nghĩa hay quan hệ ngữ nghĩa.

**5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?**

Sẽ kết hợp thêm một lớp semantic matching bên cạnh TF-IDF (hybrid search): ví dụ dùng word embedding để mở rộng query sang các từ đồng nghĩa/liên quan trước khi tính lexical overlap, hoặc dùng embedding similarity làm tín hiệu re-rank cho top-K kết quả lấy được từ TF-IDF. Ngoài ra sẽ chọn Pipeline B (Normalized) làm mặc định thay vì thử nghiệm subword thủ công, vì Part F đã chứng minh nó hiệu quả và gọn hơn Pipeline A mà không đánh đổi chất lượng.

**6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?**

AI đã được sử dụng ở những phần:
D: hỗ trợ format in ra dạng bảng của các kết quả
F/G/H: tối ưu code sử dụng trong các pần không bị lặp lại. ví dụ như tổ chức hàm pipeline() có lồng bên trong hàm search, oov, hàm evaluate để dùng được cho cả phần F và H khi có 2 relevances của 2 listr query khác nhau
