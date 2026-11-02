# LAB 07 — Bài tập tính toán

#### 6. Bài tập tính toán — Một attention head

Đề cho kích thước nhưng **không cho phần tử** của $X,W^Q,W^K,W^V$. Để tính cụ thể, chọn các ma trận minh họa sau, không coi đây là dữ kiện gốc:

$$X=\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&1&0&0\end{bmatrix},\quad
W^Q=W^K=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix},\quad
W^V=\begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}.$$

$$Q=K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}.$$

$$S=\frac{1}{\sqrt2}\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}
\approx\begin{bmatrix}.707107&0&.707107\\0&.707107&.707107\\.707107&.707107&1.414214\end{bmatrix}.$$

Softmax từng hàng: hàng đầu có mẫu số $2e^{1/\sqrt2}+1$, hàng cuối có mẫu số $2e^{1/\sqrt2}+e^{\sqrt2}$.

$$A\approx\begin{bmatrix}.401112&.197776&.401112\\.197776&.401112&.401112\\.248255&.248255&.503490\end{bmatrix},\quad
Z=AV\approx\begin{bmatrix}.401112&.197776\\.197776&.401112\\.248255&.248255\end{bmatrix}.$$

Shape: $Q,K,V:3\times2$; $S,A:3\times3$; $Z:3\times2$. Một head chính là attention ở LAB 06 sau các phép chiếu tuyến tính.

#### 7. Bài tập — Multi-head

Hai head đều có shape $T\times2$; ghép theo chiều feature được $T\times4$. Để đầu ra có chiều $d_{model}$, dùng $W^O\in\mathbb{R}^{4\times d_{model}}$.

**Vì sao cần concatenation trước khi áp dụng $W^O$?** Phép ghép giữ riêng đặc trưng của từng head; $W^O$ sau đó học cách trộn chúng và đưa về chiều cần cho residual. Cộng hai head ngay sẽ gộp các chiều tương ứng trước khi học cách phối hợp. Đây là cách viết chuẩn; về đại số có thể chia $W^O$ thành các khối rồi cộng các head đã chiếu mà không tạo tensor concat tường minh.

#### 10. Bài tập tính positional encoding

Với $d=4$, hai mẫu số là $10000^0=1$ và $10000^{1/2}=100$:

$$PE(pos)=[\sin(pos),\cos(pos),\sin(pos/100),\cos(pos/100)].$$

| pos | sin(pos) | cos(pos) | sin(pos/100) | cos(pos/100) |
|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 1 | 0.841471 | 0.540302 | 0.010000 | 0.999950 |
| 2 | 0.909297 | -0.416147 | 0.019999 | 0.999800 |

Mỗi vị trí có vector khác nhau; token giống nhau đặt ở vị trí khác nhau nhận đầu vào khác nhau.

#### 14. Bài tập tính FFN

$$x=\begin{bmatrix}1\\2\end{bmatrix},\quad W_1=I_2,\quad b_1=\begin{bmatrix}0\\-1\end{bmatrix}.$$
$$W_1x+b_1=\begin{bmatrix}1\\1\end{bmatrix},\qquad h=\mathrm{ReLU}([1,1]^T)=[1,1]^T.$$

Đề chưa cho $W_2,b_2$ nên không xác định được một giá trị số duy nhất cho $y$:

$$y=W_2\begin{bmatrix}1\\1\end{bmatrix}+b_2,\qquad y_j=(W_2)_{j1}+(W_2)_{j2}+(b_2)_j.$$

Nếu viết $W_2=\begin{bmatrix}a&b\\c&d\end{bmatrix},\ b_2=[e,f]^T$ thì $y=[a+b+e,c+d+f]^T$. Không tự gán giá trị thiếu để xem đó là đáp án của đề.
