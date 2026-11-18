# LAB 09 — Error Analysis

| Text | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- |
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | CHINA / PER | CHINA / LOC | wrong entity type | Prediction không có span khớp gold; surface xuất hiện 91 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| The former Soviet republic was playing in an Asian Cup finals tie for the first time . | Soviet / MISC | Soviet republic / LOC | wrong entity type | Prediction không có span khớp gold; surface xuất hiện 0 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar 's goal . | Bitar / PER | Bitar / LOC | wrong entity type | Prediction không có span khớp gold; surface xuất hiện 0 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| Japan , co-hosts of the World Cup in 2002 and ranked 20th in the world by FIFA , are favourites to regain their title here . | ∅ | world / MISC | spurious entity | Prediction không có span khớp gold; surface xuất hiện 0 lần trong train. Cần xét ngữ cảnh câu; tần suất thấp là giả thuyết, chưa xác định nguyên nhân. |
| Japan , co-hosts of the World Cup in 2002 and ranked 20th in the world by FIFA , are favourites to regain their title here . | FIFA / ORG | FIFA / MISC | wrong entity type | Prediction không có span khớp gold; surface xuất hiện 1 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | CHINA / PER | CHINA / LOC | wrong entity type | Gold không có span khớp prediction; surface xuất hiện 91 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| The former Soviet republic was playing in an Asian Cup finals tie for the first time . | Soviet / MISC | Soviet republic / LOC | wrong entity type | Gold không có span khớp prediction; surface xuất hiện 9 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar 's goal . | Bitar / PER | Bitar / LOC | wrong entity type | Gold không có span khớp prediction; surface xuất hiện 0 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| Bitar pulled off fine saves whenever they did . | Bitar / PER | ∅ | missing entity | Gold không có span khớp prediction; surface xuất hiện 0 lần trong train. Cần xét ngữ cảnh câu; tần suất thấp là giả thuyết, chưa xác định nguyên nhân. |
| Japan , co-hosts of the World Cup in 2002 and ranked 20th in the world by FIFA , are favourites to regain their title here . | FIFA / ORG | FIFA / MISC | wrong entity type | Gold không có span khớp prediction; surface xuất hiện 1 lần trong train. Có entity chồng lấn khác loại, gợi ý nhầm loại theo context. |
| Italy recalled Marcello Cuttitta | Marcello Cuttitta / PER | Marcello / PER | wrong boundary | Gold và prediction khác đầu mút: gold [2,4), pred [2,3). |
| Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | 1995 World Cup / MISC | World Cup / MISC | wrong boundary | Gold và prediction khác đầu mút: gold [6,9), pred [7,9). |
| Squad : Javier Pertile , Paolo Vaccari , Marcello Cuttitta , Ivan Francescato , Leandro Manteri , Diego Dominguez , Francesco Mazzariol , Alessandro Troncon , Orazio Arancio , Andrea Sgorlon , Massimo Giovanelli , Carlo Checchinato , Walter Cristofoletto , Franco Properzi Curti , Carlo Orlandi , Massimo Cuttitta , Giambatista Croci , Gianluca Guidi , Nicola Mazzucato , Alessandro Moscardi , Andrea Castellani . | Ivan Francescato / PER | Ivan / PER | wrong boundary | Gold và prediction khác đầu mút: gold [11,13), pred [11,12). |
| Squad : Javier Pertile , Paolo Vaccari , Marcello Cuttitta , Ivan Francescato , Leandro Manteri , Diego Dominguez , Francesco Mazzariol , Alessandro Troncon , Orazio Arancio , Andrea Sgorlon , Massimo Giovanelli , Carlo Checchinato , Walter Cristofoletto , Franco Properzi Curti , Carlo Orlandi , Massimo Cuttitta , Giambatista Croci , Gianluca Guidi , Nicola Mazzucato , Alessandro Moscardi , Andrea Castellani . | Ivan Francescato / PER | Francescato / PER | wrong boundary | Gold và prediction khác đầu mút: gold [11,13), pred [12,13). |
| FREESTYLE SKIING-WORLD CUP MOGUL RESULTS . | SKIING-WORLD CUP / MISC | CUP MOGUL / MISC | wrong boundary | Gold và prediction khác đầu mút: gold [1,3), pred [2,4). |

Các ví dụ được chọn có thứ tự cố định, không đại diện tỷ lệ lỗi của toàn test. Giải thích dựa vào span và tần suất quan sát được; nguyên nhân về context/hiếm cần thêm thí nghiệm.

## Context Analysis

| Surface | Gold type | Sentence | Text | Predicted spans |
| --- | --- | --- | --- | --- |
| ARIZONA | ORG | 518 | ARIZONA 6 7 0 248 332 | [(0, 1, 'ORG')] |
| ARIZONA | LOC | 545 | DALLAS AT ARIZONA | [(2, 3, 'LOC')] |
| ATLANTA | ORG | 451 | ATLANTA 10 8 .556 7 | [(0, 1, 'ORG')] |
| ATLANTA | LOC | 3295 | TORONTO AT ATLANTA | [(2, 3, 'LOC')] |
| American | MISC | 893 | In a statement Havel , who is recovering from cancer surgery , said : " Madeleine Albright is a distinguished friend , a tested diplomat , and a true American of fine origins . " | [(29, 30, 'MISC')] |
| American | ORG | 1016 | The conditions for clearance of the alliance were that British Airways and American drop 168 slots at London Heathrow airport , the busiest in Europe . | [(12, 13, 'MISC')] |

## Entity frequency

| Train frequency | Gold entities | Exact matches | Recall |
| --- | --- | --- | --- |
| Unseen (0) | 2500 | 1740 | 0.6960 |
| Rare (1–2) | 613 | 463 | 0.7553 |
| Frequent (3+) | 2535 | 2315 | 0.9132 |
