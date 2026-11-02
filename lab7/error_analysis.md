# LAB 07 — Error analysis

#### 26. Error analysis

### C1 — Dự đoán đúng, nhưng attention chủ yếu ở phần kể cốt truyện

- **Quan sát:** Model dự đoán positive, P(positive)=0.8232. Trong 64 token đầu, review đang kể về người leo núi cứu hộ và vụ cướp tiền; từ `100` ở vị trí 50 nhận attention cao nhất. Head 1 và 2 đều đặt nhiều trọng số vào token này. Từ `best` trong lời khen ở cuối review nằm tại vị trí 173, ngoài phần model nhìn thấy.
- **Diễn giải:** Nhãn đúng không chứng minh model đã đọc được lời khen kết luận. `100`, `things`, `hijacking` không phải những bằng chứng sentiment rõ ràng mà người đọc ưu tiên. Không thể dùng heatmap này để kết luận model hiểu toàn bộ đánh giá.
- **Từ hiếm/noise:** `wracked` xuất hiện một lần trong train nên bị loại bởi min_freq=2; `cloudy` không có trong train. Cả hai thành UNK. Review có câu rất dài và ít dấu câu, nhưng chưa có thí nghiệm tách riêng tác động của những yếu tố này.

### C2 — Các từ phê bình xuất hiện trong attention

- **Quan sát:** True negative, dự đoán negative; P(positive)=0.2074. Đoạn model thấy có “cheap and fake” và “poorly written”. `cheap` (vị trí 47, 0.2142) và `poorly` (60, 0.2005) là hai key nhận trọng số trung bình lớn nhất. Head 2 tập trung vào `cheap`, head 3 tập trung vào `poorly`.
- **Diễn giải:** Attention ở đây trùng với các từ phê bình mà người đọc xem là quan trọng. `laughing` cũng nằm trong top-5 nhưng trong ngữ cảnh này là phản ứng chê tác phẩm, không đơn giản là lời khen phim hài.
- **Thuật ngữ lĩnh vực:** “set decorations” và “art direction” cho biết đối tượng bị chê. Review bị cắt từ 211 còn 64 token, nhưng phần đầu đã có bằng chứng negative rõ. Không suy ra truncation luôn vô hại từ một mẫu dự đoán đúng.

### C3 — Sentiment negative rõ từ đầu, dù review rất dài

- **Quan sát:** True negative, dự đoán negative với P(positive)=0.0116. Các key `worst` (5, 0.2741), `disappointing` (50, 0.2588) và `weak` (14, 0.0468) xuất hiện trong nhóm attention cao. Head 1 nhấn `worst`; head 2 và 3 nhấn `disappointing`.
- **Diễn giải:** Những từ này phù hợp với nhận định negative của người đọc. Đoạn đầu cũng có “not only ... but ...” và “does not fight”; dự đoán đúng chưa chứng minh model xử lý đúng phạm vi phủ định, vì các từ đánh giá trực tiếp đã rất mạnh.
- **Độ dài/từ riêng:** 639 token bị cắt còn 64. `kazaf` là tên nhân vật không xuất hiện trong vocabulary train. Model vẫn đoán đúng mẫu này; không thể coi mọi UNK hoặc mọi review dài đều gây lỗi.

### E1 — Lời khen kiểu phim cult và sự chuyển ý

- **Quan sát:** Review positive nhưng dự đoán negative, P(positive)=0.1796. Đầu review nói về ấn tượng trước khi xem rồi chuyển sang “trashy, cheesy, campy and enormously fun”. `shallow` (1, 0.2174) nhận attention trung bình cao nhất; `fun` (58, 0.0391) vẫn nằm trong top-5, nhưng thấp hơn. Head 2 tập trung mạnh vào `shallow`.
- **Diễn giải:** `shallow` mô tả nhận xét ban đầu của người viết, không phải kết luận về chất lượng phim. Các từ `cheesy`, `campy` có thể đi cùng thái độ yêu thích phim cult, nên sentiment phụ thuộc ngữ cảnh và đối tượng đánh giá. Đây là giả thuyết về lỗi chuyển ý/thuật ngữ lĩnh vực, không phải bằng chứng attention ở `shallow` gây ra dự đoán.
- **Truncation:** Model thấy lời khen `fun` nhưng không thấy nhiều lời khen tiếp theo và kết luận “must-see” ở cuối review 447 token. Do vậy không nên giải thích lỗi chỉ bằng việc mọi bằng chứng positive đều bị cắt. Mẫu này không có UNK trong đoạn được đọc.

### E2 — Phủ định làm đảo nghĩa của cụm “did not stop”

- **Quan sát:** True positive, dự đoán negative, P(positive)=0.3840. Người viết bắt đầu bằng “zero interest” về thể loại âm nhạc, rồi nói “however, that did not stop me from appreciating this fantastic comedy”. `zero` (2, 0.1956), `stop` (19, 0.1380) và `sequel` (30, 0.1310) nằm trong top attention. `fantastic` ở vị trí 24 và `hilarious` ở 53 đều nằm trong input, nhưng không thuộc top-5 attention trung bình.
- **Diễn giải:** Phải phân biệt sở thích âm nhạc với đánh giá phim, đồng thời hiểu “did not stop” là phủ định việc cản trở sự yêu thích. Model có thể chưa tổng hợp tốt phạm vi phủ định và chuyển ý; heatmap chỉ cho thấy phân phối trọng số, chưa xác định nguyên nhân lỗi.
- **Domain/noise:** “This Is Spinal Tap” là tên phim dùng để so sánh; `spinal` có attention cao nhưng không nên đọc như một từ sentiment độc lập. Review có vài cấu trúc thiếu từ/sai ngữ pháp. Cả 64 token đầu không có UNK; không có căn cứ quy lỗi này cho từ ngoài vocabulary.

### E3 — Phần chê được giữ lại, phần kết luận tích cực bị cắt

- **Quan sát:** True positive, dự đoán negative, P(positive)=0.1467. Input chứa “continuity errors” và “supporting cast was really bad”; `bad` (56, 0.2194) có attention trung bình cao nhất, head 2 dành khoảng 0.8217 attention nhận trung bình cho token này. `love` (32, 0.1147) cũng xuất hiện trong top-5 nhưng nói về sự yêu thích diễn viên.
- **Diễn giải:** Review đầy đủ thừa nhận nhiều khuyết điểm rồi kết luận vẫn giải trí. Câu chuyển ý “But come on” bắt đầu tại vị trí 70 và `entertaining` nằm tại 108, đều ngoài cửa sổ 64 token. Đây là bằng chứng trực tiếp rằng model không được đọc phần chuyển ý quan trọng, dù chưa phải thử nghiệm nhân quả chứng minh chỉ cần đưa phần đuôi vào là dự đoán sẽ đúng.
- **Sentiment lẫn lộn:** Nhãn positive ở mức cả review có thể khác sắc thái của đoạn đầu. Không có UNK; lỗi này phù hợp hơn với giả thuyết truncation và đánh giá nhiều khía cạnh.

### Đối chiếu các loại lỗi theo đề

| Yếu tố | Bằng chứng trong các mẫu |
|---|---|
| Negation | E2 có “did not stop”; C3 có “not only” và “does not fight”, nhưng không thể suy ra hiểu phủ định chỉ từ đúng/sai. |
| Contrast | E1 thay đổi ấn tượng trước/sau khi xem; E2 dùng “however”; phần “But come on” của E3 bị cắt. |
| Long sequence | Cả 6 review đều dài hơn 64 token; C3 dài 639 token vẫn đoán đúng vì đầu review đã có lời chê. |
| Truncation | E3 mất phần kết luận tích cực; C1 cũng mất lời khen cuối nhưng dự đoán đúng. Truncation là rủi ro, không phải nguyên nhân duy nhất. |
| Rare word | C1: wracked (tần suất train=1), cloudy (0); C3: kazaf (0). Cả ba mẫu sai không có UNK ở phần được đọc. |
| Ambiguous sentiment | E1 dùng từ thường mang nghĩa chê trong một lời khen phim cult; E3 vừa chê kỹ thuật vừa khen tính giải trí. |
| Domain-specific term | C2: art direction; E1: campy/cheesy; E2: tên phim Spinal Tap; E3: continuity errors/supporting cast. |
| Noise | C1 có câu dài ít dấu câu; E2 có cấu trúc ngữ pháp không chuẩn. Tokenizer bỏ dấu câu; chưa đo riêng ảnh hưởng của noise. |

**Các token quan trọng có xuất hiện trong attention pattern không?** Có ở C2/C3 và một phần E1/E3, nhưng không phải lúc nào cũng khớp trực giác: C1 nhấn `100`, E2 nhấn `zero`/`stop` hơn những từ khen. **Model behavior** quan sát được là nhãn, xác suất, input đã cắt và attention weights; **human interpretation** là các giả thuyết về ngữ nghĩa, phủ định và phạm vi đánh giá. Không đồng nhất attention với explanation.

Ba mẫu sai được chọn ngẫu nhiên lần này đều là false negative; phân tích này không đại diện cho toàn bộ loại lỗi hoặc cho false positive trên test.
