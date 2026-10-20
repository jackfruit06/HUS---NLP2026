## 6. Bài tập tính toán
### Bài 1 — Một bước RNN

### Đề bài
Cho vector đầu vào tại timestep $t$:
$$x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}$$

Các ma trận trọng số:
$$W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}$$

Vector bias:
$$b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Yêu cầu tính:
1. $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$
2. $h_t = \tanh(a_t)$


### Các bước tính toán chi tiết

#### Bước 1: Nhân ma trận trọng số đầu vào với vector từ hiện tại ($W_{xh} x_t$)
$$W_{xh} x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \times 1 + 0.2 \times 0 \\ 0.1 \times 1 + 0.4 \times 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}$$

#### Bước 2: Nhân ma trận trọng số ẩn với hidden state trước đó ($W_{hh} h_{t-1}$)
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.3 \times 0.5 + 0.1 \times 0.2 \\ 0.2 \times 0.5 + 0.5 \times 0.2 \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}$$

#### Bước 3: Tính pre-activation $a_t$
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}$$

#### Bước 4: Áp dụng hàm kích hoạt phi tuyến $\tanh$
$$h_t = \tanh(a_t) = \begin{bmatrix} \tanh(0.67) \\ \tanh(0.30) \end{bmatrix}$$

Sử dụng định nghĩa $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$:
- Với $z_1 = 0.67$:
  $$e^{0.67} \approx 1.95424, \quad e^{-0.67} \approx 0.51171 \implies \tanh(0.67) = \frac{1.95424 - 0.51171}{1.95424 + 0.51171} = \frac{1.44253}{2.46595} \approx 0.58498$$
- Với $z_2 = 0.30$:
  $$e^{0.30} \approx 1.34986, \quad e^{-0.30} \approx 0.74082 \implies \tanh(0.30) = \frac{1.34986 - 0.74082}{1.34986 + 0.74082} = \frac{0.60904}{2.09068} \approx 0.29131$$

### Kết luận
$$h_t \approx \begin{bmatrix} 0.5850 \\ 0.2913 \end{bmatrix}$$



## 7. Bài 2 — Information Flow

### Đề bài
Cho sequence: `"I did not like this movie"`  
Giả sử model đọc từ trái sang phải: `"I"` $\to$ `"did"` $\to$ `"not"` $\to$ `"like"` $\to$ `"this"` $\to$ `"movie"`.

### Mô tả trạng thái ẩn $h_t$ qua từng bước:
1. $t=1$ (`"I"`): $h_1 = \tanh(W_{xh} x_{\text{"I"}} + W_{hh} h_0 + b_h)$ mang thông tin chủ ngữ.
2. $t=2$ (`"did"`): $h_2 = \tanh(W_{xh} x_{\text{"did"}} + W_{hh} h_1 + b_h)$ ghi nhận trợ động từ quá khứ.
3. $t=3$ (`"not"`): $h_3 = \tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)$ **nap thông tin phủ định (negation)** vào trạng thái ẩn.
4. $t=4$ (`"like"`): $h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)$.
5. $t=5$ (`"this"`): $h_5 = \tanh(W_{xh} x_{\text{"this"}} + W_{hh} h_4 + b_h)$.
6. $t=6$ (`"movie"`): $h_6 = \tanh(W_{xh} x_{\text{"movie"}} + W_{hh} h_5 + b_h)$ là trạng thái cuối đại diện toàn bộ câu.


Tại thời điểm model đọc `"like"`, thông tin `"not"` ảnh hưởng tới hidden state như thế nào?
- Tại $t=4$, $h_4$ nhận đầu vào trực tiếp từ vector từ $x_{\text{"like"}}$ và trạng thái ẩn trước đó $h_3$ (đã mang thông tin của từ `"not"`).
- Nhờ phép biến đổi phi tuyến và tích $W_{hh} h_3$, trọng số mạng có thể học cách chuyển hướng trạng thái biểu diễn từ sắc thái tích cực của từ `"like"` sang sắc thái tiêu cực phủ định.

**So sánh với Average Word2Vec:**
- Trong Average Word2Vec, vector của câu là:
  $$v_{\text{doc}} = \frac{1}{6} \left( v_{\text{"I"}} + v_{\text{"did"}} + v_{\text{"not"}} + v_{\text{"like"}} + v_{\text{"this"}} + v_{\text{"movie"}} \right)$$
- Từ `"like"` mang vector hướng về cực Positive, từ `"not"` là một vector độc lập. Phep cộng trung bình làm triệt tiêu lẫn nhau hoặc làm loãng phạm vi tác động của `"not"`, khiến mô hình tuyến tính rất dễ nhầm lẫn câu phủ định thành tích cực vì có sự xuất hiện của từ `"like"`.
- RNN vượt trội hơn vì có khả năng gán ghép ngữ cảnh giữa từ đứng trước và từ đứng sau.


## 8. Bài 3 — Sequence Order 

### Đề bài
Cho hai câu:
- $S_1$: `"dog bites man"` (chó cắn người — bình thường)
- $S_2$: `"man bites dog"` (người cắn chó — tin chấn động)

Giả sử cả hai câu có cùng tập từ vựng $\{ \text{"dog"}, \text{"bites"}, \text{"man"} \}$.


1. Average Word2Vec có thể phân biệt hai câu này dựa trên thứ tự hay không?
- **KHÔNG.** Phép toán của Average Word2Vec là phép cộng giao hoán:
  $$v(S_1) = \frac{1}{3} (v_{\text{dog}} + v_{\text{bites}} + v_{\text{man}}) = \frac{1}{3} (v_{\text{man}} + v_{\text{bites}} + v_{\text{dog}}) = v(S_2)$$
- Vector đại diện của $S_1$ và $S_2$ hoàn toàn trùng nhau 100%. Do đó, bất kỳ bộ phân loại nào đặt phía sau Average Word2Vec đều đưa ra cùng một dự đoán xác suất cho cả hai câu.

2. RNN có khả năng phân biệt dựa trên thứ tự như thế nào?
- RNN xử lý theo hàm đệ quy không giao hoán:
  - Với $S_1$: $h_1 = f(x_{\text{dog}})$, $h_2 = f(x_{\text{bites}}, h_1)$, $h_3 = f(x_{\text{man}}, h_2)$.
  - Với $S_2$: $h'_1 = f(x_{\text{man}})$, $h'_2 = f(x_{\text{bites}}, h'_1)$, $h'_3 = f(x_{\text{dog}}, h'_2)$.
- Vì $h_1 \neq h'_1$ nên $h_2 \neq h'_2$ và $h_3 \neq h'_3$.
- Vector trạng thái cuối $h_3$ và $h'_3$ nằm ở hai vùng không gian khác nhau, giúp mạng phân biệt rõ ràng chủ ngữ thực hiện hành động (chó cắn người hay người cắn chó).
 

## 17. Bài tập tính toán LSTM Gates 

### Đề bài
Cho một timestep cụ thể với các giá trị:
- Forget gate: $f_t = 0.9$
- Input gate: $i_t = 0.2$
- Candidate cell state: $\tilde{c}_t = 0.5$
- Cell state ở bước trước: $c_{t-1} = 0.8$

Công thức cập nhật Cell State của LSTM:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

### Tính toán
$$c_t = (0.9)(0.8) + (0.2)(0.5) = 0.72 + 0.10 = 0.82$$

### Ý nghĩa của Forget Gate
- **Khi $f_t \approx 1$:** Mô hình có xu hướng **bảo toàn nguyên vẹn** thông tin đã lưu trong cell state từ quá khứ ($c_t \approx c_{t-1}$). Về mặt giải tích, đạo hàm $\frac{\partial c_t}{\partial c_{t-1}} \approx 1$, cho phép gradient lan truyền ngược qua hàng chục hoặc hàng trăm timestep mà không bị suy giảm (giải quyết triệt để bài toán vanishing gradient).
- **Khi $f_t \approx 0$:** Mô hình chủ động **xóa bỏ hoàn toàn** ký ức cũ. Điều này hữu ích khi chuyển sang một ngữ cảnh mới, một câu mới hoặc gặp dấu ngắt đoạn trong văn bản.

 

 
