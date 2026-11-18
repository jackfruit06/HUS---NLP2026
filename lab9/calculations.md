# LAB 09 — Calculations

#### 6. Calculation Exercise 1 — BIO

| Word | BIO label |
| --- | --- |
| Elon | B-PER |
| Musk | I-PER |
| founded | O |
| SpaceX | B-ORG |
| in | O |
| California | B-LOC |
| . | O |

#### 7. Calculation Exercise 2 — Entity Boundary

`University/B-ORG of/I-ORG California/I-ORG Berkeley/I-ORG`.

**Vì sao không gán tất cả token là ORG?** Nhãn loại đơn thuần không biểu diễn điểm bắt đầu hoặc phân biệt hai organization đứng cạnh nhau. BIO xác định cả loại và ranh giới; California trong ví dụ này thuộc organization, không phải entity LOC riêng.

#### 15. Calculation Exercise 3 — Shape

B=8, T=128, d=768, C=9:

| Tensor | Shape | Ý nghĩa |
| --- | --- | --- |
| Hidden states H | (8, 128, 768) | Một vector 768 chiều cho mỗi vị trí |
| Logits Z | (8, 128, 9) | Chín logits tại mỗi vị trí |
| Labels Y | (8, 128) | Một chỉ số nhãn hoặc -100 tại mỗi vị trí |

Head có W kích thước (768,9), b kích thước (9,), tổng 6.921 tham số. Đây là số chiều giả định của bài tính; BERT-Tiny thực nghiệm dùng d=128 và head có 1.161 tham số.
