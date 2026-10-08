# Khôi phục từ notebook Colab đã tải

Nguồn: `submission/Lab22_DPO_T4_executed.ipynb` (output các cell 44, 57, 70, 73, 93, 97).
Các PNG và hai JSON trong thư mục này được trích nguyên từ output đã lưu trong notebook.

Lần chạy này dùng **200 cặp train**, **100 cặp held-out**, DPO **25 bước**. Đây là cấu hình thử nhanh,
chưa đạt yêu cầu 800 cặp train và khoảng 100 bước của bài. Tỉ lệ chosen dài hơn rejected được in là
**60,5%** (trung vị 119 so với 100 token). Notebook chỉ in **một** cặp mẫu NB2.

NB3 đã chạy xong và cell lưu adapter đã thực thi. NB4 đã sinh 8 câu cố định + 50 câu held-out,
chạy giám khảo và in `judge_summary.json`. Các file gốc trong `/content/lab22` không nằm trong
`.ipynb`; notebook không chứa đủ dữ liệu để tái tạo `side_by_side.jsonl`, `judge_results_rm.json`,
`adapter_config.json`, `split.json`, trọng số adapter, hoặc hai file Parquet của split 200 cặp.
`judge_summary.json` giữ nguyên `outputs_sha256` của `side_by_side.jsonl` gốc, nhưng hiện chưa có
file JSONL để kiểm tra hash đó.

Biểu đồ NB3 có một điểm held-out tại bước 25. Hai reward train đều dương; `rejected` cũng tăng,
nên chẩn đoán tự động `INTENDED` trong notebook cũ **không khớp hoàn toàn** với định nghĩa
“chosen tăng, rejected giảm”. Với chỉ một lần đánh giá held-out, chưa thể so xu hướng held-out
với train hay kết luận về học thuộc.
