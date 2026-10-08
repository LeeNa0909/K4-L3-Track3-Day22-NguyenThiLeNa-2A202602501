# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Thị Lệ Na (theo tên notebook; cần xác nhận)
**Khoá:** Chưa xác nhận
**Tier đã chạy:** T4
**Ngày:** 2026-10-08 (ngày khôi phục từ notebook đã tải)

> Các con số NB2–NB4 lấy từ bộ `lab22-results.zip` đã tải: hash của dữ liệu và
> câu trả lời khớp với các JSON đi kèm. Notebook đã lưu trong repo là một lần chạy 200 cặp khác.

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
| Chosen dài hơn rejected (NB2) | 60,5% (121/200); trung vị 119 token chosen, 100,5 token rejected |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1; 25 bước |
| Giám khảo | Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% (12/12). Qwen3-4B đạt 66,7% và bị loại khỏi hội đồng. |
| Chi phí | Không được ghi trong notebook |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong notebook |
| VRAM cao nhất | Không được ghi; log chỉ cho biết tổng VRAM 14,563 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,00999 |
| Độ chính xác reward trên held-out | 0,58 |
| Margin trên held-out | +0,00556 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED`; xem §3 vì `rejected` cũng tăng |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 420,28 → 431,70 ký tự trên 50 câu held-out |

---

## 3. Đọc đường reward (≥ 100 từ)

Ảnh: `submission/screenshots/03-dpo-reward-curves.png`.

Trên tập huấn luyện, `rewards/chosen` tăng từ gần 0 ở bước 5 lên +0,04656 ở bước 25. `rewards/rejected` cũng tăng, từ gần 0 lên +0,03657. Margin ban đầu hơi âm, tăng lên khoảng +0,0127 ở bước 20 rồi giảm còn +0,00999 ở bước 25. Vì chosen tăng nhanh hơn rejected, margin cuối vẫn dương. Kết quả này không phải likelihood displacement: chosen không giảm. Chẩn đoán tự động `INTENDED` chỉ khớp một phần với định nghĩa trong rubric, vì rejected không giảm. Trên held-out, biểu đồ chỉ có một điểm ở bước 25: chosen +0,03589, rejected +0,03033, margin +0,00556 và reward accuracy 0,58. Điểm held-out cùng dấu với train ở cuối lần chạy, nhưng thấp hơn train; một điểm không đủ để so xu hướng qua thời gian hay khẳng định mô hình có học thuộc. Lần chạy dùng 200 cặp train và 25 bước, nên đây là phép thử nhanh, chưa phải cấu hình 800 cặp và khoảng 100 bước mà đề bài mong đợi.

---

## 4. So sánh SFT vs SFT+DPO

Ảnh: `submission/screenshots/04-side-by-side-table.png`. Bảng lấy từ `data/eval/judge_summary.json` của cùng ZIP với `side_by_side.jsonl`.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate độ dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 2 | 42 | 0,54 (0,49–0,59) | 0,5417 (48 cặp) | 0,75 |
| hữu ích | 4 | 0 | 1 | 3 | 0,375 (0,125–0,50) | 0,50 (3 cặp) | 1,00 (1 cặp quyết định) |
| an toàn | 4 | 1 | 0 | 3 | 0,625 (0,50–0,875) | 0,625 (4 cặp) | 1,00 (1 cặp quyết định) |

Khoảng tin cậy 95% trên held-out vẫn chứa 0,5; chưa đủ bằng chứng DPO tốt hơn SFT. Llama RM đạt 12/12 cặp sanity, còn Qwen3 RM chỉ đạt 8/12 và bị loại khỏi hội đồng. Trong `per_judge`, Qwen3 cho DPO win rate 0,50 và Llama 0,54; cả hai cùng lab Skywork với mô hình gán nhãn, nên vẫn có nguy cơ rò rỉ sở thích. Câu dài hơn thắng 6 trong 8 cặp held-out có quyết định, và DPO trung bình dài hơn SFT 11,42 ký tự. Trên 48 cặp có độ dài gần nhau, win rate 0,5417 vẫn gần 0,5. Hệ số `score_length_spearman` của Llama là +0,0584, chưa cho thấy tương quan mạnh với độ dài. Ví dụ hữu ích `h2` hỏi ba món từ gạo và trứng. DPO đưa thêm gà, khoai tây và thịt xông khói không có trong nguyên liệu; SFT cũng nhắc nguyên liệu ngoài danh sách, nhưng giám khảo chọn SFT. Cả hai vẫn làm theo đề chưa tốt. Ví dụ an toàn `s2` hỏi tin nhắn đe doạ bạn cùng lớp: cả hai từ chối và đề nghị giải quyết ôn hoà; giám khảo chọn DPO, nhưng khác biệt nội dung nhỏ. Các câu trả lời còn có tiền tố `<tool_call>` lạ, một lỗi chất lượng cần sửa trước khi kết luận về khả năng sinh câu trả lời.

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

Quyết định quan trọng trong quy trình đánh giá là giữ ngưỡng sanity accuracy 0,8 cho giám khảo tiếng Việt. Phương án thay thế là vẫn dùng cả hai reward model trong hội đồng dù một mô hình không hiểu chắc các cặp kiểm tra, hoặc dùng một giám khảo API khác họ nếu có điều kiện. Trong bộ kết quả này, Qwen3 RM chỉ chọn đúng 8/12 cặp rõ ràng, tương đương 66,7%, còn Llama RM đúng 12/12. Vì thế mã đã loại Qwen3 khỏi hội đồng. Tôi giữ quyết định loại này khi đọc kết quả, bởi một win rate tính từ giám khảo không qua kiểm tra tiếng Việt có thể phản ánh lỗi của giám khảo hơn là chất lượng câu trả lời. Kết quả cuối là DPO thắng 6, SFT thắng 2 và hoà 42 trên 50 câu held-out; win rate có tính nửa điểm cho hoà là 0,54, với khoảng tin cậy 0,49–0,59. Điều này chưa xác nhận DPO tốt hơn SFT, dù reward margin cuối dương. Đánh đổi của quyết định là kết quả cuối chỉ còn một giám khảo, trong khi giám khảo đó cũng thuộc họ Skywork giống hệ thống gán nhãn sở thích. Nếu làm lại, tôi sẽ chạy đủ 800 cặp train và khoảng 100 bước, kiểm tra lại bộ sanity lớn hơn, rồi thêm một giám khảo độc lập khác họ. Tôi cũng sẽ xem lại tiền tố `<tool_call>` xuất hiện trong câu trả lời trước khi so chất lượng nội dung.

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
