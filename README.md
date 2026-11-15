# AI9

Mô hình ngôn ngữ tiếng Việt nhỏ viết bằng PyTorch, có pretrain, fine-tuning theo hướng dẫn (SFT), chat, tìm web và bộ nhớ RAG. Mô hình hiện cấu hình 46M tham số, context 768 token, tokenizer SentencePiece 16.384 token.

## Các file chính

- `model.py`: kiến trúc GPT, attention, MoE và sinh token.
- `train.py`: pretrain trên `data.txt` hoặc SFT trên dữ liệu JSONL.
- `tokenizer.py`, `preprocess.py`: chuẩn hóa dữ liệu và huấn luyện tokenizer.
- `chat.py`: chat với checkpoint, tự tìm web khi cần.
- `agent.py`, `web_search.py`, `rag.py`: định tuyến truy vấn, tìm web và lưu tài liệu đã lấy.
- `distill_generate.py`: tạo mẫu SFT bằng mô hình Qwen chạy 4-bit.
- `instruction_data.jsonl`: dữ liệu SFT hiện có.
- `distill_seed_prompts.jsonl`: câu hỏi hạt giống để sinh dữ liệu.

## Cài đặt

Cài PyTorch phù hợp với máy/GPU của bạn, rồi:

```bash
pip install sentencepiece numpy
```

Để dùng bộ tạo dữ liệu Qwen:

```bash
pip install -r requirements_distill.txt
```

Generator cần Internet để tải model Hugging Face; cấu hình mặc định dành cho GPU có tổng VRAM khoảng 2 x T4. Máy yếu hơn có thể giảm giới hạn VRAM trong `distill_generate.py` hoặc dùng model teacher nhỏ hơn qua `--model-name`.

## Huấn luyện

Pretrain cần `data.txt` và thường chạy lâu:

```bash
python train.py --mode pretrain
```

Tạo dữ liệu hướng dẫn (mặc định tối đa 1.500 mẫu; có thể tiếp tục từ output JSONL hiện có):

```bash
python distill_generate.py --num-samples 1500
```

Fine-tune. Nếu `deepseek_distill.jsonl` tồn tại, script tự ghép nó với `instruction_data.jsonl`:

```bash
python train.py --mode sft --max-steps 2000
```

Mặc định SFT tiếp tục từ `instruct_last.pth` nếu có; nếu không, bắt đầu từ `pretrain_best.pth`. Để khởi động SFT mới từ pretrained, chỉ rõ checkpoint:

```bash
python train.py --mode sft --checkpoint pretrain_best.pth
```

## Chat

```bash
python chat.py
```

Trong chat: `/web <câu hỏi>` ép tìm web, `/web off` tắt tìm web tự động, `/web on` bật lại, `/clear` xóa lịch sử hội thoại, `/forget` xóa bộ nhớ tài liệu web, `exit` thoát.

## Ghi chú

- `data.txt` hiện là dữ liệu lớn (khoảng 1,4 GB); không cần tải lại nếu đã có.
- Cần tạo tokenizer và pretrained checkpoint trước lần SFT đầu tiên.
- Các tệp ZIP và dữ liệu/checkpoint lớn bị bỏ qua bởi `.gitignore`.
