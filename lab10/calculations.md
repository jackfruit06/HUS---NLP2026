# LAB 10 — Calculations

#### 6. Calculation Exercise 1 — Start / End

Context tokens, đếm từ 0: `[The, capital, of, France, is, Paris, .]`.

| Answer | Start | End (inclusive) |
| --- | ---: | ---: |
| Paris | 5 | 5 |
| capital of France | 1 | 3 |

Các chỉ số này thuộc context token giả định, chưa có question và special tokens. Trong Python lấy `tokens[start:end+1]`.

#### 7. Calculation Exercise 2 — Start Loss

`P_start(5)=0.60`: L_start=-ln(0.60)=**0.510826**. Đề không cho riêng phân phối end, nên L_end=-ln(P_end(j)); nếu end cũng có xác suất 0.60 thì L_end=0.510826 và loss trung bình vẫn là 0.510826. Không tự coi xác suất end đã được đề cung cấp.

#### 12. Calculation Exercise 3 — Character Span

Context: `The capital of France is Paris.`

| Đại lượng | Giá trị |
| --- | --- |
| Ký tự đầu P | 25 |
| Ký tự cuối s | 29 |
| Character span half-open | [25,30) |
| Python slice | context[25:30] = Paris |
| Context-only word/token giả định | Paris, vị trí 5 |

Khi tokenize cặp question/context, token index bao gồm question và special tokens nên phải lấy từ offset_mapping, không giữ nguyên số 5. Token end là inclusive; character end dùng slicing là exclusive.


#### 19. QA F1

Gold Paris France, prediction France: overlap=1, P=1, R=0.5, F1=2/3; EM=0. Dùng Counter sau normalization, không bỏ số lần lặp như set.
