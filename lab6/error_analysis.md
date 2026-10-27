# LAB 06 — Error Analysis
> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Model:** Single-head Self-Attention + Position + Masked Mean Pooling

---

## 3. Các mẫu dự đoán đúng (Correctly Classified)

### 3.1. C1 — Positive → Positive, P(positive) = 0.9317

- **Trích đoạn model thấy:** “spoiler alert w somerset maugham classic on film about a love obsessed young man who's abused hurt and humiliated … only to find true love in the end …”
- **Nhận xét text:** Review mở bằng tóm tắt cốt truyện, có cả `classic`, `love` và các từ tiêu cực `abused`, `hurt`, `humiliated`. Không thể coi mọi từ mô tả sự kiện buồn là thái độ chê phim.
- **Attention:** Top-3 là `sensitive` (0.0517), `object` (0.0516), `unusual` (0.0463). Chúng không trùng hoàn toàn với các từ sentiment mà người đọc có thể ưu tiên như `classic`.
- **Đối chiếu hành vi:** Thay `sensitive` bằng UNK làm P(positive) **0.9317 → 0.8997**; thay `of` ở vị trí 28 làm **0.9317 → 0.9256**. Cả hai vẫn Positive.
- **Kết luận:** Model đúng dù attention không chỉ tập trung vào từ cảm xúc rõ ràng. Kết quả phù hợp với việc sử dụng nhiều tín hiệu phân tán, nhưng không chứng minh model phân biệt được cốt truyện và đánh giá. Review dài 682 token nên cũng không thể kết luận nó hiểu toàn bộ review.

### 3.2. C2 — Positive → Positive, P(positive) = 0.8892

- **Trích đoạn:** “i was stunned by this film … what an eye for setting color and detail … such visual beauty and poetry … yes beauty moves me …”
- **Nhận xét text:** Nhiều lời khen trực tiếp về hình ảnh, chi tiết và cảm xúc. `disappointed` ở gần cuối phần nhìn thấy nói về khán giả mong chờ phim hành động/soap opera, không phải kết luận chính của tác giả.
- **Attention:** Hai vị trí `beauty` nhận **0.0852** và **0.0823**, khá phù hợp trực giác về tín hiệu khen ngợi. Top tiếp theo là `hold` (0.0363), không tự mang sentiment rõ ràng.
- **Đối chiếu hành vi:** Thay một vị trí `beauty` làm P(positive) **0.8892 → 0.8744**; thay `of` ở vị trí 27 làm **0.8892 → 0.8799**. Không đổi nhãn.
- **Kết luận:** Có tín hiệu phù hợp giữa text, attention và prediction. Mức thay đổi nhỏ khi thay một `beauty` có thể do còn vị trí `beauty` thứ hai và nhiều lời khen khác; chưa đủ cơ sở gán toàn bộ prediction cho một token.

### 3.3. C3 — Negative → Negative, P(positive) = 0.0117

- **Trích đoạn:** “the twins effect is the worst movie i've seen … the support weak … the weakest vampire … your role in this movie was disappointing …”
- **Nhận xét text:** Lời chê rõ, nhiều tín hiệu cùng chiều ngay trong 100 token đầu. Không cần đọc hết 639 token để thấy bằng chứng tiêu cực ở phần đầu.
- **Attention:** `worst` cao nhất (**0.1327**), tiếp theo `too` (0.0761), `whatsoever` (0.0368).
- **Đối chiếu hành vi:** Thay `worst` làm P(positive) **0.0117 → 0.0628**; thay `seen` ở vị trí 37 làm **0.0117 → 0.0116**. Model vẫn Negative.
- **Kết luận:** Thay từ sentiment mạnh tạo thay đổi lớn hơn đối chứng thấp-attention, phù hợp hypothesis `worst` có liên quan tới prediction. Không đổi nhãn cũng phù hợp với việc có nhiều từ chê khác; không suy ra model đã hiểu mọi cấu trúc phủ định.

## 4. Các mẫu dự đoán sai (Incorrectly Classified)

### 4.1. E1 — False Positive: nhầm đối tượng lời khen trong câu so sánh

- **True:** Negative. **Pred:** Positive. **P(positive):** 0.8003.
- **Trích đoạn:** “why did they have to make such a dreadful adaptation … a huge disappointment … the 1999 movie release based on the same novel seems like a masterpiece compared to this …”
- **Phân loại lỗi:** **Ambiguous wording / contrast / phạm vi đánh giá**; có yếu tố truncation nhưng chưa đủ để quy là nguyên nhân chính.
- **Hypothesis từ text:** `masterpiece` khen bản phim năm 1999 khi so với bản đang bị chê. Một model nhạy với từ đơn lẻ có thể gán sắc thái tích cực của phim được đem ra so sánh cho review hiện tại. Ngay đầu câu đã có `dreadful adaptation`, `disappointment`, `miscast`, `no chemistry` nên thông tin tiêu cực không hoàn toàn nằm ở phần bị cắt.
- **Attention có phù hợp trực giác không?** Top key là `masterpiece` ở vị trí 56, weight **0.1233**. Tuy nhiên `dreadful` cũng đứng thứ hai (**0.0543**), vì vậy không chính xác nếu nói model hoàn toàn không attention tới lời chê. Người đọc cần xét đối tượng và cụm “compared to this”, không chỉ cực tính từ.
- **Đối chiếu model behavior:** Thay `masterpiece` tại vị trí 56 bằng UNK làm P(positive) **0.8003 → 0.4337**, đổi nhãn sang **Negative**. Thay `there's` tại vị trí 32 (weight 0.0016) làm **0.8003 → 0.7895**, vẫn Positive.
- **Kết luận:** Có bằng chứng độ nhạy với đúng vị trí `masterpiece` được quan sát trên heatmap, hỗ trợ hypothesis nhầm phạm vi lời khen. Đây chưa phải chứng minh “attention nhìn sai nên model sai”: thay token làm thay cả Q/K/V, và can thiệp UNK không giữ nguyên nghĩa câu.

### 4.2. E2 — False Negative: negation và chuyển ý

- **True:** Positive. **Pred:** Negative. **P(positive):** 0.4376.
- **Trích đoạn:** “i didn't think i would like it at all but with john ritter in it … i liked the acting was great not a lot of horrible 2nd rated comedy lines … this was a great show …”
- **Phân loại lỗi:** **Negation**, **contrast**, ngôn ngữ khẩu ngữ; không đủ bằng chứng kết luận lỗi long-range dependency.
- **Hypothesis từ text:** “didn't think … but …” chuyển từ kỳ vọng tiêu cực sang trải nghiệm tốt. `horrible` nằm trong “not a lot of horrible …”, không phải lời chê đơn giản. Review chứa nhiều `great` ngay trong phần nhìn thấy.
- **Attention:** `horrible` cao nhất (**0.1311**); ba vị trí `great` cũng nằm trong top-5 (0.0577, 0.0358, 0.0327). Phân phối có cả tín hiệu tích cực và tiêu cực, nên cần kiểm tra behavior thay vì chỉ đếm token sentiment.
- **Đối chiếu model behavior:** Thay `horrible` ở vị trí 61 làm P(positive) **0.4376 → 0.8260**, chuyển sang **Positive**. Thay `aired` ở vị trí 16 (weight 0.0027) làm **0.4376 → 0.4148**, vẫn Negative.
- **Kết luận:** Phép thử hỗ trợ hypothesis model nhạy với từ tiêu cực nhưng xử lý phủ định/chuyển ý chưa tốt. Không thể quy lỗi chỉ cho một attention weight; cần minimal pairs có kiểm soát và nhiều can thiệp hơn để kiểm tra phạm vi phủ định.

### 4.3. E3 — False Negative: plot summary, truncation và từ ngoài vocabulary

- **True:** Positive. **Pred:** Negative. **P(positive):** 0.4838, gần threshold 0.5.
- **Trích đoạn đầu:** “the gang of roses every rose has its thorns … hip hop blended perfectly together … the clothing styles the scenery and the plot are all suited …” rồi chuyển sang tóm tắt vụ cướp và các nhân vật.
- **Phần sau token 100:** “… i gave it 10 10 because the characters plot and scenery made it for me”. Model **không nhìn thấy** lời kết luận rõ này.
- **Phân loại lỗi:** **Truncation**, **ambiguous wording / plot summary**, **rare word / OOV** (5 UNK trong 100 token). OOV là quan sát về vocabulary, chưa phải bằng chứng riêng về nguyên nhân lỗi.
- **Attention:** `rose` ở vị trí 5 cao nhất (**0.1252**), một `rose` khác đứng thứ hai (0.0492); `perfectly` đứng thứ ba (0.0443). Tên/phần mô tả nội dung nhận nhiều attention hơn một tín hiệu khen rõ.
- **Đối chiếu model behavior:** Thay `rose` vị trí 5 làm P(positive) **0.4838 → 0.6715**, chuyển sang **Positive**. Thay `seven` vị trí 43 (weight 0.0021) làm **0.4838 → 0.4502**, vẫn Negative.
- **Kết luận:** Model nhạy với một từ liên quan chủ đề/tên phim; kết luận tích cực phía sau bị truncation là yếu tố có thể góp phần. Không khẳng định lỗi “100% do truncation”: câu đầu đã có “blended perfectly together”, và chưa có thí nghiệm train/evaluate lại với context dài hơn để cô lập tác động của truncation.

## 5. Tổng hợp perturbation

| Mẫu | P(positive) gốc | Thay top token → P mới | ΔP | Thay low token → P mới | ΔP | Top thay đổi nhãn? |
|---|---:|---|---:|---|---:|---|
| C1 | 0.9317 | sensitive → 0.8997 | −0.0320 | of → 0.9256 | −0.0062 | Không |
| C2 | 0.8892 | beauty → 0.8744 | −0.0148 | of → 0.8799 | −0.0092 | Không |
| C3 | 0.0117 | worst → 0.0628 | +0.0511 | seen → 0.0116 | −0.0001 | Không |
| E1 | 0.8003 | masterpiece → 0.4337 | −0.3666 | there's → 0.7895 | −0.0108 | Có |
| E2 | 0.4376 | horrible → 0.8260 | +0.3884 | aired → 0.4148 | −0.0228 | Có |
| E3 | 0.4838 | rose → 0.6715 | +0.1877 | seven → 0.4502 | −0.0336 | Có |

Trong sáu ví dụ này, thay top-attention token làm xác suất thay đổi nhiều hơn thay vị trí low-attention đối chứng. Chỉ có sáu mẫu và một đối chứng mỗi mẫu nên không suy rộng thành quy luật cho mọi token/review. `ΔP` đo độ nhạy của **toàn mô hình** khi thay input, không tách riêng phần đóng góp của attention, value và classifier.

## 6. Application — một prediction đúng và hai prediction sai

| Câu | Expected | Predicted | P(positive) | Mean-attention cao nhất |
|---|---|---|---:|---|
| I like this movie. | Positive | Negative | 0.001105 | i |
| I do not like this movie. | Negative | Negative | 0.000957 | do |
| I thought this movie would be bad, but it was actually excellent. | Positive | Negative | 0.000087 | bad |

**Mẫu đúng:** Câu thứ hai có label Negative và model dự đoán đúng. Tuy nhiên top token là `do`, còn câu positive bỏ phủ định cũng bị gán Negative. Vì vậy không dùng mẫu đúng này làm bằng chứng model hiểu `not`.

**Mẫu sai thứ nhất:** “I like this movie.” là lời khen rõ nhưng P(positive) rất thấp. Attention cao nhất tại `i` không phù hợp kỳ vọng nhấn vào `like`. Đây là quan sát đi kèm lỗi; chưa có perturbation riêng cho câu này nên không kết luận `i` gây lỗi. Hypothesis đáng kiểm tra là chênh lệch độ dài/phân phối giữa review training và câu cực ngắn, cùng đặc trưng vị trí hoặc từ vựng học được.

**Mẫu sai thứ hai:** Câu cuối đổi từ “would be bad” sang “actually excellent”, label kỳ vọng là Positive. Model tập trung mạnh nhất vào `bad` và vẫn Negative. Pattern phù hợp hypothesis yếu ở contrast, nhưng cần minimal pairs/thay token để kiểm tra trực tiếp; chưa đủ bằng chứng chỉ từ hình.

Ba câu dài lần lượt 4, 6 và 12 token, không bị truncation. Do đó **không thể dùng truncation giải thích các lỗi Application**. Biểu đồ attention cho từng câu và xác suất của nhãn dự đoán được giữ trong notebook.

## 7. Đối chiếu các nhóm lỗi trong đề

| Nhóm | Bằng chứng / phạm vi kết luận |
|---|---|
| Negation | E2 và cặp có/không `not` ở Application cho thấy cần kiểm tra phạm vi phủ định |
| Long-range dependency | Chưa có phép thử cô lập độ dài dependency; review dài không tự chứng minh lỗi loại này |
| Ambiguous wording | E1 có so sánh hai phiên bản phim; E3 xen plot summary với đánh giá |
| Sarcasm | Chưa có mẫu rõ đủ để gán nhãn chắc chắn trong 6 ví dụ được chọn |
| Rare word | E3 có 5 UNK; cần kiểm tra embedding/vocabulary riêng để quy tác động |
| Domain shift | Application là câu ngắn tự viết, khác phân phối review; đây là hypothesis về distribution shift, không phải thay hẳn miền sentiment |
| Truncation | Cả 6 review dài hơn 100; E3 có lời kết luận tích cực bị cắt, nhưng chưa cô lập hiệu ứng |
| Noise | E2 có khẩu ngữ/cách viết không chuẩn; chưa chạy normalization ablation nên không quy lỗi riêng cho noise |

**Giới hạn:** Mẫu đã được chọn theo đúng/sai nên không đại diện tần suất lỗi toàn test set. Phân loại lỗi là nhận định thủ công, có thể chồng lấn. Các can thiệp chỉ kiểm tra độ nhạy cục bộ, không được dùng để chọn lại checkpoint hoặc tối ưu trên test.
