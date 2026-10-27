"""LAB 06 — Scaled dot-product attention, cài đặt từ đầu bằng NumPy.

Chạy: python attention_numpy.py
Q: (..., n_query, d_k), K: (..., n_key, d_k), V: (..., n_key, d_v).
mask=True giữ key; mask=False loại key. Mỗi query phải còn ít nhất một key.
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax ổn định; cho phép -inf để biểu diễn vị trí bị mask."""
    x = np.asarray(x, dtype=np.float64)
    maximum = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - maximum)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def attention(Q, K, V, *, scaled=True, temperature=1.0, mask=None):
    """Trả về (scores sau scaling/temperature/mask, weights, context)."""
    Q, K, V = [np.asarray(a, dtype=np.float64) for a in (Q, K, V)]
    # Kiểm tra temperature
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature phải là số hữu hạn lớn hơn 0.")

    # Kiểm tra số chiều và kích thước tensor
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError("Q, K, V phải có ít nhất 2 chiều.")

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Chiều cuối của Q và K phải bằng nhau.")

    if K.shape[-2] != V.shape[-2]:
        raise ValueError("Số key của K phải bằng số vector của V.")

    if Q.shape[:-2] != K.shape[:-2] or K.shape[:-2] != V.shape[:-2]:
        raise ValueError("Các batch dimensions của Q, K, V phải tương thích.")

    if not all(np.all(np.isfinite(a)) for a in (Q, K, V)):
        raise ValueError("Q, K, V phải chứa các giá trị hữu hạn.")
    
    scores = Q @ np.swapaxes(K, -1, -2)
    scores = scores / (np.sqrt(Q.shape[-1]) if scaled else 1.0) / temperature
    if mask is not None:
        mask = np.asarray(mask)
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("Mask không tương thích với scores.") from exc
        # Mỗi query phải còn ít nhất một key được giữ lại
        if not np.all(mask.any(axis=-1)):
            raise ValueError("Mỗi query phải còn ít nhất một key.")

        scores = np.where(np.broadcast_to(mask, scores.shape), scores, -np.inf)
    weights = softmax(scores)
    return scores, weights, weights @ V


def scaled_dot_product_attention(Q, K, V, *, temperature=1.0, mask=None):
    """S = QK^T / sqrt(d_k) / temperature; A = softmax(S); C = AV."""
    return attention(Q, K, V, scaled=True, temperature=temperature, mask=mask)


def entropy(weights, axis=-1):
    """Shannon entropy dùng log tự nhiên (đơn vị: nat), quy ước 0 log 0 = 0."""
    weights = np.asarray(weights, dtype=np.float64)
    terms = np.zeros_like(weights)
    positive = weights > 0
    terms[positive] = weights[positive] * np.log(weights[positive])
    return -terms.sum(axis=axis)


def verify_attention():
    """Kiểm tra đáp án tính tay, stability, batch và padding mask."""
    Q = np.array([[1., 0.]])
    K = np.array([[1., 0.], [0., 1.], [1., 1.]])
    V = np.array([[1., 0.], [0., 2.], [2., 1.]])
    scores, weights, context = attention(Q, K, V, scaled=False)
    np.testing.assert_allclose(scores, [[1, 0, 1]])
    np.testing.assert_allclose(weights, [[0.4223187983, 0.1553624035, 0.4223187983]])
    np.testing.assert_allclose(context, [[1.266956395, 0.733043605]], rtol=1e-8)
    np.testing.assert_allclose(softmax([1000, 1001, 1002]), softmax([1, 2, 3]))
    np.testing.assert_allclose(softmax([1, 1, 1, 1]), np.full(4, .25))
    _, masked, result = scaled_dot_product_attention(Q, K, V, mask=[[True, False, True]])
    np.testing.assert_allclose(masked, [[.5, 0, .5]])
    np.testing.assert_allclose(result, [[1.5, .5]])
    qb, kb, vb = [np.stack([a, a]) for a in (Q, K, V)]
    _, wb, cb = scaled_dot_product_attention(qb, kb, vb)
    np.testing.assert_allclose(wb.sum(-1), 1)
    np.testing.assert_allclose(cb[0], scaled_dot_product_attention(Q, K, V)[2])
    for kwargs in ({'temperature': 0}, {'mask': [[False, False, False]]}):
        try:
            scaled_dot_product_attention(Q, K, V, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Input không hợp lệ phải bị từ chối.')
    print('PASSED: tính tay, stable softmax, uniform scores, mask, batch, invalid inputs.')
    return scores, weights, context


if __name__ == '__main__':
    s, a, c = verify_attention()
    print('Scores:', s)
    print('Weights:', a)
    print('Context:', c)
