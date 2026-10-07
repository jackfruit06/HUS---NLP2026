# Error Analysis — LAB 03


## A. Ba similarity đúng

### A1. doctor → surgeon, dentist, ophthalmologist, physician, veterinarian

- **Observed:** top-5 của `doctor` là surgeon (0.721), dentist (0.711), ophthalmologist (0.696), physician (0.689), veterinarian (0.675) — toàn bộ là nghề y.
- **Expected:** các từ chỉ người làm nghề y, đặc biệt là physician (đồng nghĩa).
- **Possible explanation:** các từ này **có ngữ cảnh giống nhau** (người ta "consult / ask / visit" một bác sĩ, nha sĩ, bác sĩ phẫu thuật…), nên Skip-gram học được vector gần nhau. Chúng gần nhau không phải vì luôn đứng cạnh nhau mà vì **thay thế được cho nhau** trong cùng kiểu câu.
- **Evidence from corpus:**
  - `top_contexts("doctor")` = eye, should, talk, care, health, consult, order, go, skin, visit;
    `top_contexts("surgeon")` = lasik, **eye**, you're, **should**, surgery, plastic, **ask**, best, said, expert → chung "eye", "should", hỏi/tư vấn.
  - 6 câu chứa cả doctor và surgeon, đều là quảng cáo LASIK: *"speak to your present eye **doctor** … receive a referral to an eye **surgeon**"*.
  - Tần suất đủ lớn để học: doctor = 263, surgeon = 117, physician = 86, dentist = 49.

### A2. disease → infection, chronic, autoimmune, hypertension, artery

- **Observed:** top-5 của `disease` là infection (0.760), chronic (0.760), autoimmune (0.750), hypertension (0.748), artery (0.746).
- **Expected:** các từ về bệnh tật.
- **Possible explanation:** "disease" xuất hiện trong ngữ cảnh y khoa rất nhất quán (tên bệnh, nguy cơ, điều trị), nên vector của nó nằm trong cụm từ y khoa. Quan hệ ở đây là **cùng chủ đề** (bệnh và loại bệnh / bộ phận liên quan), không phải đồng nghĩa.
- **Evidence from corpus:**
  - `top_contexts("disease")` = heart, risk, disease, patients, control, lyme, coronary, **chronic**, cause…;
    `top_contexts("autoimmune")` = diseases, disorder, **disease**, immune, disorders, **chronic**… → chia sẻ ngữ cảnh "chronic", "disease".
  - "coronary", "heart" là context hàng đầu của disease → giải thích "artery", "hypertension" (bệnh tim mạch).
  - 3 câu chứa cả disease và infection: *"the treatment of gum **disease** repair of decay and the elimination of **infection**"*.
  - disease = 368 lần, infection = 94 lần.

### A3. computer → computers, desktop, laptop, defraggler, phone's

- **Observed:** top-5 của `computer` là computers (0.681), desktop (0.645), laptop (0.644), defraggler (0.643), phone's (0.627).
- **Expected:** laptop, desktop, device, software.
- **Possible explanation:** computer, desktop, laptop đều xuất hiện trong ngữ cảnh thiết bị/công nghệ (phần mềm, pin, mạng, di động). "defraggler" (phần mềm chống phân mảnh ổ đĩa) vào top vì các trang hướng dẫn sửa máy tính nhắc nó cùng ngữ cảnh với computer — đúng chủ đề, dù là tên riêng hiếm.
- **Evidence from corpus:**
  - `top_contexts("computer")` = software, surge, protector, science, image, training, **laptop**…;
    `top_contexts("laptop")` = **computer**, gaming, battery, phone, **desktop**, mobile…
  - 10 câu chứa cả computer và laptop, 8 câu chứa cả computer và desktop: *"whether it is the **desktop computer** where microsoft's windows has about 90 percent market share"*.
  - computer = 622, desktop = 106, laptop = 138 lần.

---

## B. Ba similarity sai / bất ngờ

### B1. football → replica, shirt, shirts, thcheap, shirtr

- **Observed:** top-5 của `football` là replica (0.885), shirt (0.832), shirts (0.798), thcheap (0.734), shirtr (0.680) — toàn áo bóng đá và **từ rác** (thcheap, shirtr).
- **Expected:** soccer, match, team, player, league.
- **Possible explanation:** **noisy data + domain bias**. Một số trang spam SEO bán áo bóng đá giả lặp lại "cheap football shirts" hàng trăm lần, chiếm phần lớn số lần "football" xuất hiện, nên vector của football bị kéo về phía "áo". Các token "thcheap", "shirtr" là chữ bị dính liền trong chính trang spam.
- **Evidence from corpus:**
  - football xuất hiện 573 lần trong 113 documents, nhưng chỉ **4 documents spam** (chứa replica/thcheap) đã chiếm **348 lần (≈ 61%)**.
  - `top_contexts("football")` = football, **replica, shirt, shirts**, united, kingdom, kits, **cheap, thcheap**, emperor.
  - 69 câu chứa cả football và replica: *"**thcheap football shirts cheap football shirts** true"*, *"**replica football shirtir** elders"*.

### B2. banana → sprouts, beef, smoothie, almond, chilli

- **Observed:** top-5 của `banana` là sprouts (0.706), beef (0.704), smoothie (0.697), almond (0.695), chilli (0.690) — nguyên liệu nấu ăn, không có trái cây nào.
- **Expected:** apple, mango, fruit, orange.
- **Possible explanation:**
  - **Frequency (nguyên nhân chính):** banana chỉ xuất hiện 42 lần, và các láng giềng còn hiếm hơn (sprouts = 8, smoothie = 6, chilli = 6 lần). Với ít dữ liệu như vậy, vector của các từ này được cập nhật rất ít lần, nên vị trí của chúng trong không gian vector **gần như do nhiễu** — chúng gần nhau mà không có ngữ cảnh chung thực sự.
  - **Domain:** banana trong web text thường nằm trong ngữ cảnh đồ ăn (banana bread, choco banana, công thức xay sinh tố) chứ không phải trong ngữ cảnh "các loại quả", nên cả cụm láng giềng đều là đồ ăn.
  - **Polysemy nhẹ:** "banana peppers" là một loại **ớt**, không phải chuối.
- **Evidence from corpus:**
  - `top_contexts("banana")` = festivals, export, **bread**, pulp, board, programme, **pan**, green.
  - Ngữ cảnh chung gần như **không có**: banana–sprouts: không có; banana–chilli: không có; banana–smoothie: chỉ "green"; banana–almond: chỉ "cake".
  - **0 câu** chứa cả banana và sprouts / smoothie / chilli. Câu duy nhất có banana và almond: *"put all ingredients in the blender starting with the **banana** add fruit then the **almond** butter on top"*.
  - → Khác với A1–A3 (có nhiều ngữ cảnh chung), similarity ở đây **không có bằng chứng ngữ cảnh** hỗ trợ — dấu hiệu của vector kém tin cậy do từ hiếm.
  - Câu có "banana peppers": *"green peppers **banana peppers** peppadews portabello mushrooms"*.

### B3. patient → patient's, patients

- **Observed:** hai từ gần `patient` nhất là patient's (0.690) và patients (0.647) — thực chất là **cùng một từ ở dạng khác**.
- **Expected:** các từ khác nghĩa nhưng liên quan, như doctor, nurse, hospital, treatment.
- **Possible explanation:** **preprocessing / vocabulary limitation**. Tokenizer giữ nguyên `'s` (mẫu `[a-z0-9]+(?:'[a-z]+)?`) và không đưa từ về dạng gốc (không lemmatization), nên patient, patients, patient's là ba token riêng với ba vector riêng. Vì chúng có ngữ cảnh gần như giống hệt nhau nên luôn chiếm đầu bảng, đẩy các từ liên quan thật sự xuống dưới. Hiện tượng tương tự: computer → computers, phone's (A3).
- **Evidence from corpus:**
  - patient = 320, patients = 547, patient's = 27 lần.
  - `top_contexts("patient")` = care, including, outcomes, records, health, medical…;
    `top_contexts("patients")` = care, cancer, medical, heart, disease, treatment… → chung "care", "medical".
  - 12 câu chứa cả hai dạng: *"vent **patients** require highly skilled respiratory therapists beyond the average oxygen **patient**"*.

---

## C. Tổng hợp nguyên nhân

| Ví dụ | Nguyên nhân chính |
|---|---|
| football → replica, thcheap | noisy data, domain bias (spam SEO) |
| banana → sprouts, smoothie, chilli | frequency thấp (vector gần như nhiễu, không có ngữ cảnh chung), domain đồ ăn, polysemy ("banana peppers") |
| patient → patient's, patients | preprocessing (giữ `'s`, không lemmatization) |

Các ví dụ đúng (A1–A3) đều có điểm chung: từ xuất hiện **đủ nhiều** (hàng trăm lần) và trong **ngữ cảnh nhất quán**. Các ví dụ sai đều do dữ liệu — từ hiếm, dữ liệu nhiễu, hoặc cách tiền xử lý — chứ không phải do thuật toán Word2Vec.

> Lưu ý: similarity (số trong ngoặc) lấy từ lần chạy notebook gần nhất; vì `workers=4` nên mỗi lần train lại có thể lệch nhẹ (riêng từ hiếm như banana có thể đổi hẳn láng giềng). Các số liệu corpus (tần suất, context, số câu) không đổi.
