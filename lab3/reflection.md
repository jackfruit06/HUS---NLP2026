# Reflection — Từ Word2Vec đến Transformer

## 1. Bảng tổng kết

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không | Sparse | Không (vector biểu diễn document, mỗi từ chỉ là một chiều cố định) |
| Co-occurrence | Không | Sparse | Không |
| Word2Vec | Không| Dense | Không |
| Contextual embedding | Có | Dense | Có (mỗi lần xuất hiện, tùy câu, có một vector riêng) |

