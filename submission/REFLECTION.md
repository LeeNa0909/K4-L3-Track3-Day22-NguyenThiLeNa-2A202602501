# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Thị Lệ Na (theo tên notebook; cần xác nhận)
**Khoá:** Chưa xác nhận
**Tier đã chạy:** T4
**Ngày:** 2026-10-08 (ngày khôi phục từ notebook đã tải)

> Các con số NB3–NB4 được khôi phục từ output của notebook Colab đã tải;
> xem `recovered_colab/RECOVERY.md` về những tệp gốc không còn.

---

## NB0. Vì sao margin tăng khi chosen giảm?

DPO tối ưu chênh lệch `β[(pc − rc) − (pr − rr)]`, không tối ưu riêng xác suất của `chosen`.
Nếu log-xác suất `chosen` giảm 3 nat nhưng `rejected` giảm 5 nat so với reference,
margin vẫn tăng `2β`. Đây là **likelihood displacement**: mô hình phân biệt hai câu
rõ hơn, dù bản thân câu được chọn cũng ít có khả năng được sinh ra hơn.

---

## NB2. Đọc cặp mẫu đã lưu trong notebook

Notebook đã tải chỉ in một cặp mẫu trên split 200 cặp. Câu hỏi là cách sao chép và chia sẻ
văn bản. `chosen` hướng dẫn chi tiết cho máy tính và điện thoại, nhưng dài hơn và thêm
gợi ý chụp màn hình/dịch vụ ngoài không cần thiết. `rejected` ngắn hơn và đủ ý chính,
song lẫn ký tự lạ. Vì vậy nhãn `chosen` có thể hợp lý về độ rõ ràng, nhưng độ dài cũng
có thể ảnh hưởng. Không có bằng chứng trong notebook để khẳng định đã đọc đủ ba cặp
mẫu của **chính split Colab này**.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4, 14,563 GB hiển thị trong log |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · **200** huấn luyện / 100 held-out; không trùng câu hỏi |
| Chosen dài hơn rejected (NB2) | 60,5% (121/200); trung vị 119 token chosen, 100 token rejected |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1; 25 bước |
| Giám khảo | Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% (12/12). Qwen3-4B đạt 66,7% và bị loại khỏi hội đồng. |
| Chi phí | Không được ghi trong notebook |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong notebook |
| VRAM cao nhất | Không được ghi; log chỉ cho biết tổng VRAM 14,563 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,00455 |
| Độ chính xác reward trên held-out | 0,53 |
| Margin trên held-out | +0,00835 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED`; xem §3 vì `rejected` cũng tăng |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 496,16 → 498,88 ký tự trên 50 câu held-out |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh khôi phục: `recovered_colab/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Trên tập huấn luyện, `rewards/chosen` tăng từ khoảng +0,004 ở bước 5 lên +0,04755
ở bước 25. `rewards/rejected` cũng tăng từ khoảng +0,002 lên +0,04300; nó **không giảm**.
Margin train dương, đạt khoảng +0,013 ở bước 20 rồi lùi về +0,00455 ở bước 25.
Do đó nhãn tự động `INTENDED` trong lần chạy này chỉ phản ánh rằng chosen dương và
margin dương; nó không khớp hoàn toàn với định nghĩa “chosen tăng, rejected giảm”.
Đây cũng không phải likelihood displacement, vì chosen không giảm. Đánh giá held-out
chỉ xuất hiện **một điểm** ở bước 25: chosen +0,04214, rejected +0,03379, margin
+0,00835 và reward accuracy 0,53. Điểm này cho thấy margin dương tại cuối lần chạy,
nhưng không đủ để biết đường held-out có đi theo đường train qua thời gian hay không;
chưa thể kết luận có hoặc không có học thuộc. Lần chạy chỉ dùng 200 cặp và 25 bước,
thấp hơn yêu cầu 800 cặp và khoảng 100 bước, nên kết quả này là phép thử nhanh.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh khôi phục: `recovered_colab/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 2 | 45 | 0,51 (0,47–0,55) | 0,51 (50 cặp) | 0,80 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0,50 (0,50–0,50) | 0,50 (4 cặp) | Không xác định: không có cặp thắng/thua |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 0,375 (0,125–0,50) | 0,375 (4 cặp) | 1,00 (chỉ 1 cặp quyết định) |

Giám khảo được dùng: Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 1,00 ·
`score_length_spearman` trên held-out: −0,0559.

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Khoảng tin cậy 95% trên held-out chứa 0,5, vì vậy chưa đủ bằng chứng rằng DPO tốt
hơn SFT. Qwen3 RM chỉ đạt 8/12 cặp sanity (66,7%) và bị loại; Llama đạt 12/12.
`per_judge` cho DPO win rate 0,53 ở Qwen3 và 0,51 ở Llama. Chênh lệch 0,02 nhỏ,
nhưng Qwen3 không qua kiểm tra tiếng Việt và cả hai vẫn thuộc họ Skywork, nên
không thể loại trừ rò rỉ sở thích. Trong 5 cặp held-out có quyết định, câu dài hơn
thắng 4 cặp; tuy nhiên cả 50 cặp đều thuộc nhóm độ dài gần bằng nhau, win rate
của nhóm này vẫn chỉ 0,51, và DPO trung bình chỉ dài hơn 2,72 ký tự. Không có
bằng chứng chắc rằng DPO cải thiện nhờ viết dài. Ví dụ hữu ích `h2` hỏi món từ
gạo và trứng, nhưng cả SFT và DPO đều gợi ý gà/cá không có trong nguyên liệu:
đây là lỗi làm theo chỉ dẫn và hai bản gần như giống nhau. Ví dụ an toàn `s2`
yêu cầu viết tin nhắn đe doạ: cả hai đều từ chối và gợi ý giải quyết hoà bình;
DPO đổi cách diễn đạt nhưng chưa thể gọi là tiến bộ rõ ràng. Output in trong
notebook bị rút gọn, nên nhận xét hai ví dụ chỉ dựa trên phần văn bản còn lưu.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng trong quy trình đánh giá là giữ ngưỡng sanity accuracy
0,8 cho giám khảo tiếng Việt. Phương án thay thế là vẫn dùng cả hai reward model
trong hội đồng dù một mô hình không hiểu chắc các cặp kiểm tra, hoặc dùng một
giám khảo API khác họ nếu có điều kiện. Trong notebook này, Qwen3 RM chỉ chọn
đúng 8/12 cặp rõ ràng, tương đương 66,7%, còn Llama RM đúng 12/12. Vì thế mã
đã loại Qwen3 khỏi hội đồng. Tôi giữ quyết định loại này khi đọc kết quả, bởi
một win rate tính từ giám khảo không qua kiểm tra tiếng Việt có thể phản ánh lỗi
của giám khảo hơn là chất lượng câu trả lời. Kết quả cuối là DPO thắng 3, SFT
thắng 2 và hoà 45 trên 50 câu held-out; win rate có tính nửa điểm cho hoà là
0,51, với khoảng tin cậy 0,47–0,55. Điều này không xác nhận DPO tốt hơn SFT,
dù reward margin cuối dương. Đánh đổi của quyết định là kết quả cuối chỉ còn
một giám khảo, trong khi giám khảo đó cũng thuộc họ Skywork giống hệ thống gán
nhãn sở thích. Nếu làm lại, tôi sẽ chạy đủ 800 cặp train và khoảng 100 bước,
kiểm tra lại bộ sanity lớn hơn, rồi thêm một giám khảo độc lập khác họ. Khi đó
tôi mới tin hơn vào sự khác biệt win rate giữa hai mô hình.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
