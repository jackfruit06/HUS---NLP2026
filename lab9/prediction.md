# LAB 09 — Prediction

#### 12. Prediction Before Experiment

| Prediction | Reason | Expected evidence |
| --- | --- | --- |
| Pretrained Transformer có thể đạt entity F1 cao hơn baseline. | Context nhiều vị trí và biểu diễn pretrained có thể hỗ trợ phân biệt loại và boundary; baseline chỉ học các đặc trưng từ/cửa sổ cục bộ. | Precision/Recall/F1 entity trên cùng toàn bộ test set, kèm chi phí train. |
| Entity nhiều từ thường khó khớp hoàn toàn hơn entity một từ. | Chỉ cần sai một đầu mút hoặc một nhãn là cả entity không khớp. Tuy vậy entity dài có thể có dấu hiệu từ vựng rõ hơn. | F1 theo nhóm 1,2,3,4+ words ở mục 22. |
| Entity hiếm hoặc chưa thấy có thể bị bỏ sót nhiều hơn. | Baseline khó ghi nhớ tên; pretrained có thể suy luận theo context nhưng vẫn chịu ảnh hưởng cách viết và domain. | Recall entity theo tần suất surface form trong train; chỉ tính tần suất trên train. |
