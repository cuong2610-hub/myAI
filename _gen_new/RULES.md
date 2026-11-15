# QUY TẮC SINH DỮ LIỆU SFT CHO AI9

Bạn sinh dữ liệu huấn luyện tiếng Việt cho một mô hình ngôn ngữ nhỏ (context 768 token).
Mỗi dòng là một cặp hỏi–đáp tự chứa, dùng được ngay để SFT.

## Định dạng bắt buộc

- File JSONL, mỗi dòng đúng một object: `{"user": "...", "assistant": "..."}`
- Chỉ 2 khóa `user` và `assistant`, không thêm khóa khác.
- UTF-8, không BOM, kết thúc file bằng newline.
- Mọi xuống dòng bên trong chuỗi phải escape thành `\n` (JSON hợp lệ).
- Nếu viết bằng script Python: `json.dumps(obj, ensure_ascii=False)`.

## Độ dài (quan trọng: model chỉ có 768 token)

- `user`: 8–45 từ (khoảng 15–180 ký tự).
- `assistant`: 15–75 từ (khoảng 50–350 ký tự).
- Tổng `user` + `assistant` ≤ 900 ký tự. Không có ngoại lệ.
- Code snippet (nếu có): tối đa 6 dòng, viết trong cùng chuỗi với `\n`.

## Văn phong câu hỏi

- Tiếng Việt tự nhiên, giữ thuật ngữ kỹ thuật tiếng Anh (RoPE, KV cache, pointer, hash table...).
- Đa dạng cấu trúc, luân phiên các kiểu:
  - định nghĩa có ngữ cảnh ("Trong transformer, ...")
  - so sánh ("A khác B ở điểm nào khi ...")
  - nguyên nhân ("Vì sao ... xảy ra?")
  - khi nào dùng / chọn cái nào
  - tính toán ngắn, dự đoán output
  - tìm lỗi trong đoạn code/đoạn lập luận
  - viết hàm nhỏ, tối ưu, đọc hiểu code
- TUYỆT ĐỐI KHÔNG dùng các tiền tố máy móc lặp lại:
  - "Hãy giải thích ngắn gọn:"
  - "Trả lời câu hỏi sau:"
  - "Cho một giải thích dễ hiểu về:"
  - "Tóm tắt khái niệm này:"
  - "Tại sao hoặc như thế nào?"
  - "(mẫu N)" hoặc bất kỳ đánh số mẫu
- Không mở đầu mọi câu bằng "Bạn có thể cho biết". Tối đa 2 câu trong cả file dùng cụm đó.

## Văn phong câu trả lời

- Trả lời trực tiếp, đúng trọng tâm, 1–4 câu. Không lan man, không chào hỏi, không nhắc lại câu hỏi.
- Nêu lý do/điều kiện/đánh đổi khi câu hỏi cần giải thích.
- Với bài tính: nêu cách làm ngắn và kết quả cuối, phải tự kiểm tra lại số.
- Với code: code chạy được, ngắn, kèm 1 câu giải thích nếu cần.
- Không bịa định luật, số liệu, tên tài liệu. Nếu không chắc, chọn câu hỏi khác.

## Chống trùng (bắt buộc)

- Đọc `existing_topics.txt`: đó là toàn bộ chủ đề file cũ đã có. Không hỏi lại các ý đó, kể cả diễn đạt khác.
- Trong cùng file của bạn: không lặp câu hỏi, không hỏi cùng một khái niệm 2 lần.
- Không dùng lại cùng một mẫu câu cho nhiều mục (ví dụ 20 câu đều "X là gì?").
- Mỗi mục phải kiểm tra được bằng một ý riêng biệt, không phải bản diễn đạt của mục khác.

## Quy trình

1. Đọc `RULES.md` (file này), `existing_topics.txt`, và phần spec của bạn trong `SPECS.md`.
2. Sinh đúng số lượng yêu cầu.
3. Ghi ra đúng đường dẫn output được giao.
4. Tự kiểm tra bằng Python: đếm dòng, parse JSON, kiểm tra khóa, độ dài, câu hỏi trùng lặp.
5. Nếu sai, sửa file rồi kiểm tra lại. Chỉ kết thúc khi tất cả kiểm tra đạt.
6. Trả lời cuối: số dòng, số câu hỏi trùng (phải là 0), và 2 ví dụ đại diện.

## Lệnh kiểm tra gợi ý

```
python -c "import json,sys; rows=[json.loads(l) for l in open(r'<file>',encoding='utf-8') if l.strip()]; qs=[r['user'].strip().lower() for r in rows]; print(len(rows), len(set(qs)), max(len(r['user'])+len(r['assistant']) for r in rows)); assert all(set(r)=={'user','assistant'} for r in rows)"
```
