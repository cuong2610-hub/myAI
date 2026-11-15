# MANIFEST — 2.000 mẫu SFT bổ sung (2026-11-15)

## Kết quả

| Nhóm | Yêu cầu | Đã sinh | Batch nguồn |
|---|---:|---:|---|
| AI / Machine Learning / LLM | 400 | 400 | A1, A2, A3, A4 |
| C++, thuật toán, cấu trúc dữ liệu | 500 | 500 | B1, B2, B3, B4, B5 |
| Python, lập trình | 300 | 300 | C1, C2, C3 |
| Toán, Lý, Hóa | 300 | 300 | D1, D2, D3 |
| Hội thoại nhiều lượt | 300 | 300 | E1, E2, E3 |
| Suy luận, giải thích, xử lý vấn đề | 200 | 200 | F1, F2 |
| **Tổng** | **2.000** | **2.000** | 20 batch x 100 |

Phân bổ chi tiết từng batch: xem `SPECS.md`.

## Định dạng

- Schema: `{"user": "...", "assistant": "..."}` — giống hệt dữ liệu cũ, `train.py` đọc trực tiếp.
- UTF-8, không BOM, LF, mỗi dòng một object JSON hợp lệ.
- Hội thoại nhiều lượt: `user` chứa 2-3 lượt trước với nhãn `Người dùng:` / `Trợ lý:`,
  `assistant` là câu trả lời cho lượt cuối. Cùng kiểu với các mẫu hội thoại đã có trong file cũ.

## Kiểm tra đã chạy

- 2.000/2.000 dòng parse JSON thành công, đúng 2 khóa, không rỗng.
- 0 câu hỏi trùng tuyệt đối giữa 20 batch.
- 0 cặp gần trùng (Jaccard token >= 0,80).
- 0 câu trùng/na ná 276 chủ đề đã có trong `instruction_data.jsonl` cũ (Jaccard >= 0,78).
- Độ dài: trung bình 244 ký tự/mục, dài nhất 639 ký tự (giới hạn context 768 token).
- Không còn tiền tố máy móc kiểu "Hãy giải thích ngắn gọn:", "(mẫu N)".

## Gộp file

- Bản gốc được sao lưu: `instruction_data.backup-2026-11-15.jsonl` (845 dòng).
- `instruction_data.jsonl` hiện có **2.845 dòng** = 845 cũ + 2.000 mới, xen kẽ theo nhóm
  (mỗi 20 dòng liên tiếp phủ đủ 20 batch) để batch huấn luyện nào cũng đa dạng.

## Ghi chú về dữ liệu cũ (chưa sửa)

- 845 dòng cũ chỉ chứa 276 chủ đề; 155 chủ đề bị lặp 2-6 lần bằng cách đổi tiền tố câu hỏi
  (khoảng 568 dòng dư nếu gộp theo chủ đề).
- 104 dòng có câu trả lời rỗng nghĩa kiểu "Đây là khái niệm quan trọng trong AI...".
- 122 câu còn dấu vết sinh tự động "(mẫu N)".
- Đề xuất: lọc trùng theo chủ đề + bỏ 104 dòng placeholder trước khi SFT. Chưa thực hiện vì
  cần bạn xác nhận.

## File nguồn

- `out/A1..F2.jsonl` — 20 file, mỗi file đúng 100 dòng, giữ nguyên để tái tạo/kiểm tra.
- `RULES.md` — quy tắc sinh dữ liệu.
- `SPECS.md` — phạm vi từng batch.
- `existing_topics.txt` — 276 chủ đề cũ dùng để chống trùng.
