# LAB 10 — Error Analysis

| Question | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- |
| What was the name of the leader through the Great Depression and World War II? | James Bryant Conant | 1900. James Bryant Conant | wrong start | Gold chars [899,918), pred [893,918); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| All Italian chocolate is made from what alone? | cocoa butter | butter | wrong start | Gold chars [2007,2019), pred [2013,2019); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 4 window; mọi window đều được xét. |
| What was Temüjin' sister's name? | Temülen | Hachiun, and Temüge, and one sister named Temülen | wrong start | Gold chars [82,89), pred [40,89); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| Which logo had the DW Tardis insignia removed? | The logo for the Twelfth Doctor | Twelfth Doctor | wrong start | Gold chars [933,964), pred [950,964); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 2 window; mọi window đều được xét. |
| Who holds an annual International Arts Fair in Newcastle? | NewcastleGateshead | late August bank holiday weekend, is an annual two-day multicultural event, blending drama, music and food from Punjabi, Pakistani, Bengali and Hindu cultures. NewcastleGateshead | wrong start | Gold chars [188,206), pred [28,206); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| When did Martin Luther publish his translation of the New Testament? | 1522 | 1522, and he and his collaborators completed the translation of the Old Testament in 1534 | wrong end | Gold chars [68,72), pred [68,157); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| What are BSkyB's standard definition broadcasts compliant with? | DVB-compliant MPEG-2 | DVB-compliant MPEG-2, with the Sky Movies and Sky Box Office channels | wrong end | Gold chars [46,66), pred [46,115); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| What do the Treaties generally limit? | free trade | free trade, to prioritise other values such as public health, consumer protection, labour rights, fair competition, and environmental improvement | wrong end | Gold chars [1975,1985), pred [1975,2120); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 3 window; mọi window đều được xét. |
| The hardest problems in NP can be analogously written as what class of problems? | NP-complete | NP-complete problems contains the most difficult problems in NP, in the sense that they are the ones most likely not to be in P. Because the problem P = NP is not solved, being able to reduce a known NP-complete problem, Π2, to another problem, Π1 | wrong end | Gold chars [244,255), pred [244,491); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| The use of what device represented the last major evolution of the steam engine? | steam turbines | steam turbines starting in the late part of the 19th century. Steam turbines are generally more efficient than reciprocating piston type steam engines | wrong end | Gold chars [68,82), pred [68,218); có phần giao nhau, phù hợp lỗi partial answer/boundary. Context tạo 1 window; mọi window đều được xét. |
| What did John Dunn invent? | keyed Northumbrian smallpipes | the city | wrong span | Gold chars [508,537), pred [475,483); span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân. Context tạo 1 window; mọi window đều được xét. |
| When was Warsaw ranked as the 32nd most liveable city in the world? | 2012 | Warsaw | wrong span | Gold chars [3,7), pred [47,53); span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân. Context tạo 1 window; mọi window đều được xét. |
| What would the local council of Newcastle like to link their local bike networks to? | national networks | National Cycling strategy, Newcastle first developed its cycling strategy in 1998 | wrong span | Gold chars [810,827), pred [32,113); span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân. Context tạo 1 window; mọi window đều được xét. |
| What properties are analyzed with a conoscopic lens by petrologists?  | birefringence, pleochroism, twinning, and interference properties | a conoscopic lens. In the electron microprobe | wrong span | Gold chars [483,548), pred [554,599); span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân. Context tạo 1 window; mọi window đều được xét. |
| What year was Jamukha surrendered to Temüjin? | 1206 | 1201, a khuruldai elected Jamukha as Gür Khan, "universal ruler", a title used by the rulers of the Qara Khitai | wrong span | Gold chars [657,661), pred [213,324); span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân. Context tạo 1 window; mọi window đều được xét. |

## Wrong Start vs Wrong End

| # wrong start | # wrong end | # both boundaries wrong |
| --- | --- | --- |
| 445 | 449 | 386 |

Số đếm so với mọi annotated character span và có thể chồng nhau. Vị trí khác nhưng answer text giống gold vẫn có thể được EM chấp nhận. Chọn 15 mẫu EM=0 theo thứ tự test cố định để minh họa, không suy tỷ lệ lỗi từ mẫu chọn. Context dài không đồng nghĩa truncation: mọi window đều được đánh giá; coverage được báo cáo riêng.

## EM = 0, F1 > 0

| Question | Gold | Prediction | EM | F1 | Explanation |
| --- | --- | --- | --- | --- | --- |
| What would the local council of Newcastle like to link their local bike networks to? | national networks \| national networks \| to national networks | National Cycling strategy, Newcastle first developed its cycling strategy in 1998 | 0.0000 | 0.1538 | Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0. |
| What was the name of the leader through the Great Depression and World War II? | James Bryant Conant \| James Bryant Conant \| James Bryant Conant | 1900. James Bryant Conant | 0.0000 | 0.8571 | Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0. |
| When did Martin Luther publish his translation of the New Testament? | 1522 \| 1522 \| 1522 | 1522, and he and his collaborators completed the translation of the Old Testament in 1534 | 0.0000 | 0.1429 | Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0. |
| All Italian chocolate is made from what alone? | cocoa butter \| cocoa butter \| cocoa butter | butter | 0.0000 | 0.6667 | Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0. |
| When did Kenya become an independent country? | 1963 \| Kenya Independence Act \| 1963 | Kenya | 0.0000 | 0.5000 | Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0. |

## Controlled context changes

| Condition | Samples | EM | F1 |
| --- | --- | --- | --- |
| Original | 50 | 12.0000 | 22.9931 |
| Question-word distractor | 50 | 12.0000 | 21.5148 |
| Reversed sentences | 50 | 16.0000 | 25.4843 |

Giữ gold text nhưng đảo câu có thể làm đổi quan hệ tham chiếu, nên đây là độ nhạy với phép biến đổi chứ chưa cô lập nguyên nhân. Model không được huấn luyện hoặc chọn theo các phép biến đổi này.
