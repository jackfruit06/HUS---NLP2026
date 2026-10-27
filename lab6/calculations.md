# LAB 06 — Bài tập tính toán Attention
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026

Quy ước: mỗi token là một hàng khi viết ma trận; một query được viết dạng vector trong phần tính tay. Softmax tính theo các key. Làm tròn ở kết quả cuối, không làm tròn ở các bước trung gian.

## 8. Bài 1 — Attention scores

### Đề bài

$$q=\begin{bmatrix}1\\2\end{bmatrix},\qquad K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

### Các bước tính toán

$$s_1=q^Tk_1=1\times1+2\times0=1.$$
$$s_2=q^Tk_2=1\times0+2\times1=2.$$
$$s_3=q^Tk_3=1\times1+2\times1=3.$$

### Kết luận

$$\boxed{s=[1,2,3]}.$$

Key thứ ba có dot product lớn nhất. Dot product đo mức tương thích trong không gian biểu diễn và phụ thuộc cả độ lớn vector; không nên đồng nhất nó với cosine similarity.

## 9. Bài 2 — Softmax attention weights

### Đề bài

Cho $s=[1,2,3]$, tính $\alpha=\operatorname{softmax}(s)$.

### Bước 1: Tính hàm mũ và mẫu số

$$e^1\approx2.718282,\quad e^2\approx7.389056,\quad e^3\approx20.085537.$$
$$Z=e^1+e^2+e^3\approx30.192875.$$

### Bước 2: Chuẩn hóa

$$\alpha_1=\frac{2.718282}{30.192875}\approx0.090031.$$
$$\alpha_2=\frac{7.389056}{30.192875}\approx0.244728.$$
$$\alpha_3=\frac{20.085537}{30.192875}\approx0.665241.$$

$$\boxed{\alpha\approx[0.090031,0.244728,0.665241]}.$$

### Kiểm tra và giải thích

$$\sum_i\alpha_i=\frac{e^1+e^2+e^3}{e^1+e^2+e^3}=1.$$

Hàm mũ tăng đơn điệu, còn mẫu số chung dương nên score lớn nhất nhận weight lớn nhất. Với ví dụ ở mục 6 của đề, $s=[2,1,0]$ cho $\alpha\approx[0.665241,0.244728,0.090031]$; token đầu nhận weight lớn nhất.

## 10. Bài 3 — Context vector

### Đề bài

$$\alpha=[0.1,0.2,0.7],\quad v_1=\begin{bmatrix}1\\0\end{bmatrix},\quad v_2=\begin{bmatrix}0\\1\end{bmatrix},\quad v_3=\begin{bmatrix}2\\2\end{bmatrix}.$$

### Tính weighted sum

$$c=0.1v_1+0.2v_2+0.7v_3
=\begin{bmatrix}0.1\\0\end{bmatrix}+\begin{bmatrix}0\\0.2\end{bmatrix}+\begin{bmatrix}1.4\\1.4\end{bmatrix}
=\boxed{\begin{bmatrix}1.5\\1.6\end{bmatrix}}.$$

**Value đóng góp nhiều nhất:** $v_3$ có weight 0.7 và weighted contribution $[1.4,1.4]^T$, lớn nhất ở cả hai tọa độ trong ví dụ. Nói chung, chỉ nhìn weight chưa đủ kết luận mức đóng góp vì còn phụ thuộc độ lớn/hướng của value.

## 11. Bài 4 — Toàn bộ attention computation

### Đề bài

$$q=\begin{bmatrix}1\\0\end{bmatrix},\quad K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad V=\begin{bmatrix}1&0\\0&2\\2&1\end{bmatrix}.$$

### Bước 1: Scores (chưa scaling)

$$s=[1\times1+0\times0,\;1\times0+0\times1,\;1\times1+0\times1]=[1,0,1].$$

### Bước 2: Attention weights

$$Z=e+1+e=2e+1\approx6.436564.$$
$$\alpha=\left[\frac e{2e+1},\frac1{2e+1},\frac e{2e+1}\right]\approx[0.422319,0.155362,0.422319].$$

### Bước 3: Context

$$c=0.422319\begin{bmatrix}1\\0\end{bmatrix}
+0.155362\begin{bmatrix}0\\2\end{bmatrix}
+0.422319\begin{bmatrix}2\\1\end{bmatrix}.$$
$$\boxed{c\approx\begin{bmatrix}1.266956\\0.733044\end{bmatrix}}.$$

Key 1 và key 3 nhận weight bằng nhau vì có score bằng nhau, nhưng value khác nhau nên contribution khác nhau. Bài này dùng dot-product attention theo yêu cầu, không tự thêm hệ số scaling.

## 14. Bài 5 — Scaling

### Đề bài

Cho $d_k=4$, $QK^T=[4,2,0]$.

### Bước 1: Chia căn dimension

$$\sqrt{d_k}=2,\qquad S=\frac{[4,2,0]}2=[2,1,0].$$

### Bước 2: So sánh softmax

| Trường hợp | Scores | Attention weights | Max weight |
|---|---|---|---:|
| Không scaling | [4, 2, 0] | [0.866813, 0.117310, 0.015876] | 0.866813 |
| Scaling $1/\sqrt4$ | [2, 1, 0] | [0.665241, 0.244728, 0.090031] | 0.665241 |

**Nhận xét:** Scaling giữ thứ tự scores nhưng giảm chênh lệch, khiến softmax ít tập trung hơn; entropy tăng từ khoảng 0.441057 lên 0.832396 nat. Việc giảm score quá lớn giúp tránh softmax quá bão hòa trong điều kiện thường gặp; không có nghĩa scaling luôn làm classifier chính xác hơn.

## 15. Bài 6 — Temperature

Giữ $s=[1,2,3]$, thay $\tau$ trong $\operatorname{softmax}(s/\tau)$. Dùng $\tau$ thay vì T để tránh nhầm với sequence length.

$$H(\alpha)=-\sum_i\alpha_i\ln\alpha_i.$$

| Temperature | Weights | Max weight | Entropy (nat) | Nhận xét |
|---:|---|---:|---:|---|
| 0.5 | [0.015876, 0.117310, 0.866813] | 0.866813 | 0.441057 | Tập trung mạnh nhất |
| 1.0 | [0.090031, 0.244728, 0.665241] | 0.665241 | 0.832396 | Mức gốc |
| 2.0 | [0.186324, 0.307196, 0.506480] | 0.506480 | 1.020191 | Phẳng hơn |

Với score hữu hạn, $\tau\to\infty$ cho phân phối đều. Khi $\tau\to0^+$, trọng số tập trung vào vị trí score cực đại; nếu nhiều vị trí cùng cực đại thì chúng chia đều phần trọng số trong giới hạn.

## 17. Bài 7 — Numerical stability

Với $x=[1000,1001,1002]$, tính trực tiếp $e^{1002}$ có thể overflow kiểu float64.

$$m=\max(x)=1002,\qquad x-m=[-2,-1,0].$$
$$\operatorname{softmax}(x)=\frac{[e^{-2},e^{-1},1]}{e^{-2}+e^{-1}+1}
\approx[0.090031,0.244728,0.665241].$$

Phép trừ max không đổi kết quả vì:

$$\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
=\frac{e^{-m}e^{x_i}}{e^{-m}\sum_j e^{x_j}}
=\frac{e^{x_i}}{\sum_j e^{x_j}}.$$V

Code mask key bằng $-\infty$ cho weight 0, nhưng không chấp nhận hàng toàn $-\infty$ vì khi đó softmax không xác định.

## 21. Bài 8 — Self-attention: “I like cats”

### Giả định bổ sung

Đề nêu kích thước X và yêu cầu các ma trận WQ/WK/WV nhưng không ghi số cụ thể. Để hoàn thành phép tính, **tự chọn**:

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
W_K=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
W_V=\begin{bmatrix}1&0\\1&1\end{bmatrix}.$$

Ba hàng tương ứng `I`, `like`, `cats`. Đây là ví dụ số học, không phải pretrained embeddings.

### Bước 1: Tạo Q, K, V

$$Q=XW_Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$
$$K=XW_K=\begin{bmatrix}1&1\\0&1\\1&2\end{bmatrix},\qquad
V=XW_V=\begin{bmatrix}1&0\\1&1\\2&1\end{bmatrix}.$$

Ví dụ hàng cuối của K: $[1,1]W_K=[1\times1+1\times0,\;1\times1+1\times1]=[1,2]$.

### Bước 2: Scores

$$QK^T=\begin{bmatrix}1&0&1\\1&1&2\\2&1&3\end{bmatrix}.$$
$$S=\frac{QK^T}{\sqrt2}\approx
\begin{bmatrix}0.707107&0&0.707107\\0.707107&0.707107&1.414214\\1.414214&0.707107&2.121320\end{bmatrix}.$$

### Bước 3: Softmax theo từng hàng

Ví dụ hàng đầu: $\alpha_{1,:}=[e^{1/\sqrt2},1,e^{1/\sqrt2}]/(2e^{1/\sqrt2}+1)$.

$$A\approx\begin{bmatrix}
0.401112&0.197776&0.401112\\
0.248255&0.248255&0.503490\\
0.283995&0.140029&0.575975
\end{bmatrix}.$$

Mỗi hàng có tổng bằng 1 trước khi làm tròn; các hàng không bắt buộc bằng nhau, ma trận không bắt buộc đối xứng.

### Bước 4: Contextual representations

$$C=AV\approx\begin{bmatrix}
1.401112&0.598888\\
1.503490&0.751745\\
1.575975&0.716005
\end{bmatrix}.$$

Ví dụ $C_{1,:}=0.401112[1,0]+0.197776[1,1]+0.401112[2,1]\approx[1.401112,0.598888]$.

**Ý nghĩa:** representation của mỗi token là tổ hợp value của nhiều token trong cùng sequence. Các con số này minh họa sự tương tác, không chứng minh quan hệ chủ ngữ–động từ–tân ngữ.

## 25, 29. Bài 9 — Chi phí full self-attention

| Sequence length T | Attention matrix | Số phần tử | Bộ nhớ một matrix float32 |
|---:|---|---:|---:|
| 10 | 10 × 10 | 100 | 400 bytes |
| 100 | 100 × 100 | 10,000 | 40,000 bytes |
| 500 | 500 × 500 | 250,000 | 1,000,000 bytes |
| 1,000 | 1,000 × 1,000 | 1,000,000 | 4,000,000 bytes |
| 10,000 | 10,000 × 10,000 | 100,000,000 | 400,000,000 bytes |

Tăng T từ 10 lên 1000: T tăng 100 lần, số score tăng $100^2=10,000$ lần. Bộ nhớ bảng trên chỉ tính **một sample, một head, một matrix**; training còn lưu nhiều tensor và gradient. Projection có chi phí $O(Td^2)$; phần full attention $QK^T$ và $AV$ có chi phí $O(T^2d)$ với dimensions cùng bậc d. Với một query cố định, số score là T và chi phí chỉ tuyến tính theo T khi giữ dimension cố định.
