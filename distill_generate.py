
"""
Qwen3-14B 4-bit -> AI9 synthetic SFT dataset generator.

Thiết kế cho Kaggle:
- 2 x Tesla T4 14.6 GB
- Qwen3-14B 4-bit NF4
- device_map="auto"
- Giới hạn VRAM mỗi GPU để tránh OOM khi materialize weights
- Không dùng DDP cho teacher: một model được shard trên cả 2 GPU
- Resume từ file JSONL hiện có
- In Q/A trực tiếp trong quá trình sinh
- Lọc câu trả lời quá ngắn/dài/repetitive
- Deduplicate user question
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# QUAN TRỌNG:
# Đặt allocator config trước khi import torch.
# ----------------------------------------------------------------------
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import argparse
import gc
import json
import random
import re
import time
import traceback
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ======================================================================
# CONFIG
# ======================================================================

DEFAULT_MODEL = "Qwen/Qwen3-14B"

DEFAULT_SEEDS = "distill_seed_prompts.jsonl"
DEFAULT_BASE_DATA = "instruction_data.jsonl"
DEFAULT_OUTPUT = "qwen3_distill.jsonl"

DEFAULT_NUM_SAMPLES = 1500

# Giới hạn VRAM để tránh GPU bị full trong lúc Transformers materialize
# các tensor.
GPU_MEMORY = "13GiB"

# CPU RAM dùng làm vùng dự phòng nếu device_map cần offload.
CPU_MEMORY = "48GiB"

# 4-bit NF4
COMPUTE_DTYPE = torch.float16


# ======================================================================
# NORMALIZATION
# ======================================================================

def normalize_space(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_key(text: str) -> str:
    text = normalize_space(text).lower()

    # Bỏ wrapper ở đầu câu nếu dataset cũ có.
    patterns = [
        r"^giải thích dễ hiểu\s*:\s*",
        r"^hãy trả lời chính xác và tự nhiên\s*:\s*",
        r"^hãy giải từng bước rồi kết luận\s*:\s*",
        r"^phân tích câu hỏi sau và kiểm tra kết quả\s*:\s*",
        r"^hãy nêu cách làm và đáp án cuối cùng cho\s*:\s*",
        r"^một học sinh hỏi\s*:\s*",
        r"^câu hỏi\s*:\s*",
        r"^question\s*:\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_text = re.sub(pattern, "", text, count=1)
            if new_text != text:
                text = new_text.strip()
                changed = True

    return text


# ======================================================================
# LOAD DATA
# ======================================================================

def load_seed_prompts(path: str) -> list[str]:
    if not os.path.exists(path):
        return []

    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON lỗi {path}:{line_no}")
                continue

            prompt = str(obj.get("prompt", "")).strip()

            if prompt:
                result.append(prompt)

    return result


def load_base_questions(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []

    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(obj.get("user", "")).strip()

            if question:
                result.append({"question": question, "category": str(obj.get("category", "general"))})

    return result


def load_existing(path: str):
    """
    Đọc dataset đã có.

    Trả về:
        rows
        seen_questions
    """

    rows = []
    seen_questions = set()

    if not os.path.exists(path):
        return rows, seen_questions

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON lỗi ở output dòng {line_no}")
                continue

            user = normalize_space(obj.get("user", ""))
            assistant = normalize_space(obj.get("assistant", ""))

            if not user or not assistant:
                continue

            key = normalize_key(user)

            if not key:
                continue

            # Deduplicate câu hỏi cũ.
            if key in seen_questions:
                continue

            obj["user"] = user
            obj["assistant"] = assistant

            rows.append(obj)
            seen_questions.add(key)

    return rows, seen_questions


# ======================================================================
# QUESTION POOL
# ======================================================================

def build_source_pool(seeds: list[str], base: list[dict]):
    """
    Gộp seed + base questions rồi loại duplicate.
    """

    pool = []
    seen = set()

    sources = [{"question": q, "category": infer_category(q)} for q in seeds] + base
    for item in sources:
        q = item["question"]
        q = normalize_space(q)

        if not q:
            continue

        key = normalize_key(q)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        category = normalize_space(item.get("category", "general")).lower()
        if category == "general":
            category = infer_category(q)
        pool.append({"question": q, "category": category or "general"})

    return pool


def infer_category(question: str) -> str:
    if re.search(r"code|lập trình|python|program|thuật toán", question, re.I):
        return "code"
    if re.search(r"toán|tính|phương trình|xác suất|diện tích|chu vi", question, re.I):
        return "math"
    return "vietnamese"


# ======================================================================
# QUESTION VARIATION
# ======================================================================

QUESTION_WRAPPERS = [
    "{q}",
    "Bạn có thể giải thích {q} không?",
    "Mình muốn hỏi: {q}",
    "Cho mình hỏi {q}",
    "Giải thích giúp mình {q}",
    "Làm thế nào để {q}",
    "Mình chưa hiểu, {q} là gì?",
    "Có thể hướng dẫn mình về {q} không?",
]


def make_tasks(
    source_pool: list[dict],
    count: int,
    seed: int,
):
    """
    Tạo task từ source pool.

    Không dùng wrapper một cách máy móc cho toàn bộ dataset.
    Ưu tiên câu gốc, wrapper chỉ dùng khi cần thêm mẫu.
    """

    if not source_pool:
        raise RuntimeError("Không có source question.")

    rng = random.Random(seed)

    tasks = []

    # Trộn source.
    shuffled = source_pool[:]
    rng.shuffle(shuffled)

    buckets = {}
    for item in shuffled:
        buckets.setdefault(item["category"], []).append(item)
    shuffled = []
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:
                shuffled.append(bucket.pop())

    # Nếu target <= số source thì mỗi source chỉ dùng một lần.
    if count <= len(shuffled):
        return shuffled[:count]

    # Nếu target lớn hơn source pool, tạo thêm biến thể.
    seen = set()

    # Trước tiên lấy toàn bộ câu gốc.
    for q in shuffled:
        key = normalize_key(q["question"])

        if key in seen:
            continue

        seen.add(key)
        tasks.append(q)

        if len(tasks) >= count:
            return tasks

    # Sau đó dùng wrapper.
    max_attempts = count * 20
    attempts = 0

    while len(tasks) < count and attempts < max_attempts:
        attempts += 1

        item = rng.choice(source_pool)
        q = item["question"]
        wrapper = rng.choice(QUESTION_WRAPPERS)

        candidate = normalize_space(
            wrapper.format(q=q)
        )

        key = normalize_key(candidate)

        if key in seen:
            continue

        seen.add(key)
        tasks.append({"question": candidate, "category": item["category"]})

    return tasks


# ======================================================================
# OUTPUT CLEANING
# ======================================================================

def clean_output(text: str) -> str:
    text = str(text)

    # Xóa thinking của Qwen nếu có.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.I | re.S,
    )

    # Một số wrapper thường gặp.
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>", 1)[1]
        text = text.split("</answer>", 1)[0]

    text = re.sub(
        r"^Đáp án\s*[:：]\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^Trả lời\s*[:：]\s*",
        "",
        text,
        flags=re.I,
    )

    # Không để code fence thừa.
    text = text.replace("```text", "")
    text = text.replace("```plaintext", "")
    text = text.replace("```", "")

    text = normalize_space(text)

    return text.strip()


# ======================================================================
# QUALITY FILTER
# ======================================================================

def looks_bad_answer(
    answer: str,
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str]:

    answer = answer.strip()

    if len(answer) < min_chars:
        return True, f"quá ngắn ({len(answer)} ký tự)"

    if len(answer) > max_chars:
        return True, f"quá dài ({len(answer)} ký tự)"

    # Quá nhiều dòng thường là model bị loop.
    if answer.count("\n") > 100:
        return True, "quá nhiều dòng"

    # Kiểm tra lặp nguyên dòng.
    lines = [
        x.strip().lower()
        for x in answer.splitlines()
        if x.strip()
    ]

    if len(lines) >= 8:
        counts = {}

        for line in lines:
            counts[line] = counts.get(line, 0) + 1

        repeated = max(counts.values())

        if repeated >= 5:
            return True, "lặp dòng"

    # Kiểm tra một đoạn ngắn bị lặp rất nhiều lần.
    compact = re.sub(r"\s+", " ", answer.lower())

    words = compact.split()

    if len(words) >= 30:
        for n in (3, 4, 5):
            grams = {}

            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                grams[gram] = grams.get(gram, 0) + 1

            if grams and max(grams.values()) >= 8:
                return True, "lặp cụm từ"

    return False, ""


def looks_self_contradictory(answer: str) -> bool:
    """Catch explicit opposing claims; this is a conservative text heuristic."""
    text = normalize_space(answer).lower()
    patterns = (
        (r"\bkhông đúng\b", r"\bđúng là\b"),
        (r"\bkhông thể\b", r"\bcó thể\b"),
        (r"\bkhông phải\b", r"\blà\b"),
        (r"\bsai\b", r"\bchính xác\b"),
    )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for i, first in enumerate(sentences):
        for second in sentences[i + 1:]:
            if any(re.search(a, first) and re.search(b, second) or re.search(b, first) and re.search(a, second)
                   for a, b in patterns):
                shared = set(re.findall(r"\w+", first)) & set(re.findall(r"\w+", second))
                if len(shared) >= 2:
                    return True
    return False


def semantic_duplicate(answer: str, previous: list[str], threshold=0.92) -> bool:
    """Conservative near-duplicate check based on normalized word overlap."""
    words = set(re.findall(r"\w+", normalize_key(answer)))
    if len(words) < 8:
        return False
    for old in previous:
        other = set(re.findall(r"\w+", normalize_key(old)))
        if len(other) >= 8 and len(words & other) / len(words | other) >= threshold:
            return True
    return False


# ======================================================================
# QWEN PROMPT
# ======================================================================

SYSTEM_PROMPT = """
Bạn là giáo viên và người biên soạn dữ liệu huấn luyện cho một mô hình
ngôn ngữ tiếng Việt nhỏ.

Nhiệm vụ:
- Trả lời câu hỏi của người dùng bằng tiếng Việt.
- Chính xác, tự nhiên, rõ ràng.
- Không học vẹt câu hỏi.
- Không nhắc rằng bạn đang tạo dataset.
- Không nói về prompt, system prompt hay quá trình sinh dữ liệu.
- Không tự bịa nguồn hoặc số liệu.
- Nếu câu hỏi thiếu thông tin, nói rõ phần còn thiếu.
- Với toán: trình bày các bước tính cần thiết rồi kết luận.
- Với lập trình: giải thích ngắn gọn và đưa code đúng khi cần.
- Với kiến thức: giải thích phù hợp với học sinh.
- Không cần viết <think>.
- Chỉ xuất câu trả lời cuối cùng.
""".strip()


def build_inputs(
    tokenizer,
    question: str,
    input_device: torch.device,
    max_input_tokens: int,
):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    if getattr(tokenizer, "chat_template", None):
        try:
            result = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )

            return result.to(input_device)

        except Exception as exc:
            print(
                "[WARN] chat_template lỗi, dùng fallback:",
                type(exc).__name__,
            )

    fallback = (
        "Hệ thống:\n"
        + SYSTEM_PROMPT
        + "\n\nNgười dùng:\n"
        + question
        + "\n\nTrợ lý:\n"
    )

    result = tokenizer(
        fallback,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    return result.input_ids.to(input_device)


# ======================================================================
# GENERATION
# ======================================================================

def generate_answer(
    model,
    tokenizer,
    input_device,
    question: str,
    max_new_tokens: int,
    max_input_tokens: int,
):
    input_ids = build_inputs(
        tokenizer=tokenizer,
        question=question,
        input_device=input_device,
        max_input_tokens=max_input_tokens,
    )

    attention_mask = torch.ones_like(input_ids)

    eos_id = tokenizer.eos_token_id

    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = eos_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,

            max_new_tokens=max_new_tokens,

            do_sample=True,

            # Nhiệt độ thấp để dataset ổn định hơn.
            temperature=0.35,

            top_p=0.90,

            repetition_penalty=1.05,

            eos_token_id=eos_id,
            pad_token_id=pad_id,

            use_cache=True,
        )

    generated = output_ids[
        0,
        input_ids.shape[-1]:,
    ]

    raw = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )

    answer = clean_output(raw)

    return answer, raw


# ======================================================================
# MODEL LOADING
# ======================================================================

def load_teacher(model_name: str):
    print()
    print("=" * 70)
    print("Loading teacher")
    print("=" * 70)

    print("Model :", model_name)
    print("Quant : 4-bit NF4")
    print("Device: auto")
    print("VRAM  :", GPU_MEMORY, "per GPU")
    print("CPU   :", CPU_MEMORY)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA không khả dụng. Hãy bật GPU trên Kaggle."
        )

    gpu_count = torch.cuda.device_count()

    print()
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)

        print(
            f"GPU {i}: {props.name} | "
            f"{total_gb:.1f} GB"
        )

    if gpu_count < 2:
        print(
            "[WARN] Không thấy 2 GPU. "
            "Script vẫn thử chạy với GPU hiện có."
        )

    # --------------------------------------------------------------
    # Hugging Face token
    # --------------------------------------------------------------

    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print("HF_TOKEN: đã nạp")
    else:
        print(
            "HF_TOKEN: chưa nạp "
            "(không bắt buộc nếu model tải được)"
        )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # --------------------------------------------------------------
    # BitsAndBytes
    # --------------------------------------------------------------

    try:
        import bitsandbytes as bnb

        print(
            "bitsandbytes:",
            bnb.__version__,
        )

    except Exception as exc:
        raise RuntimeError(
            "Không import được bitsandbytes. "
            "Hãy chạy: pip install -U 'bitsandbytes>=0.46.1'"
        ) from exc

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=COMPUTE_DTYPE,

        bnb_4bit_use_double_quant=True,

    )

    # --------------------------------------------------------------
    # Device map
    # --------------------------------------------------------------

    max_memory = {}

    for i in range(gpu_count):
        max_memory[i] = GPU_MEMORY

    max_memory["cpu"] = CPU_MEMORY

    print()
    print("max_memory:")
    print(max_memory)

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------
    #
    # QUAN TRỌNG:
    # Không dùng device_map="balanced".
    #
    # balanced có thể nhồi một GPU sát 14.6 GB trong quá trình
    # materialize tensor -> OOM.
    #
    # auto + max_memory cho phép Transformers chừa VRAM và
    # chuyển phần cần thiết sang GPU còn lại / CPU.
    # --------------------------------------------------------------

    print()
    print("Đang load model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,

        quantization_config=bnb_config,

        device_map="auto",

        max_memory=max_memory,

        low_cpu_mem_usage=True,

        trust_remote_code=True,

        token=hf_token,

    )

    model.eval()

    # --------------------------------------------------------------
    # Xác định device của input embedding.
    #
    # KHÔNG dùng:
    #     next(model.parameters()).device
    #
    # vì model đang được sharded trên nhiều GPU.
    # --------------------------------------------------------------

    input_device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    print()
    print("Model loaded.")
    print("Input device:", input_device)

    # In device map để kiểm tra model thực sự được chia.
    if hasattr(model, "hf_device_map"):
        print()
        print("Device map:")

        for name, device in model.hf_device_map.items():
            print(
                f"  {name}: {device}"
            )

    return model, tokenizer, input_device


# ======================================================================
# SAVE DATASET
# ======================================================================

def save_dataset(
    path: str,
    rows: list[dict],
):
    temp_path = Path(
        str(path) + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temp_path.replace(path)


def evaluate_dataset(path: str):
    """Write deterministic, category-specific checks; not a factuality judge."""
    groups = {"code": [], "math": [], "vietnamese": [], "other": []}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(row.get("user", ""))
            a = str(row.get("assistant", ""))
            category = str(row.get("category", "")).lower()
            if category not in groups or category == "general":
                category = infer_category(q)
            groups[category].append({"line": line_no, "user": q, "assistant": a})
    report = {"counts": {k: len(v) for k, v in groups.items()}, "checks": {}}
    for category, rows in groups.items():
        if category == "other":
            continue
        report["checks"][category] = {
            "samples": len(rows),
            "empty_answer": sum(not r["assistant"].strip() for r in rows),
            "self_contradiction_flags": sum(looks_self_contradictory(r["assistant"]) for r in rows),
            "code_fence_samples" if category == "code" else "answer_samples": [
                {"line": r["line"], "user": r["user"], "assistant": r["assistant"][:500]}
                for r in rows[:20]
            ],
        }
    report_path = str(path) + ".evaluation.json"
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluation report: {report_path} | counts={report['counts']}")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--seed-file",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--base-data",
        default=DEFAULT_BASE_DATA,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1536,
    )

    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument("--evaluate", action="store_true", help="Tạo báo cáo review riêng code/toán/tiếng Việt rồi thoát.")

    args = parser.parse_args()

    if args.evaluate:
        evaluate_dataset(args.output)
        return

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 70)
    print("Dataset")
    print("=" * 70)

    seeds = load_seed_prompts(
        args.seed_file
    )

    base = load_base_questions(
        args.base_data
    )

    source_pool = build_source_pool(
        seeds,
        base,
    )

    existing, seen_questions = load_existing(
        args.output
    )

    print(
        f"Seed prompts  : {len(seeds)}"
    )

    print(
        f"Base questions : {len(base)}"
    )

    print(
        f"Unique sources : {len(source_pool)}"
    )

    print(
        f"Đã có         : {len(existing)}"
    )

    print(
        f"Mục tiêu      : {args.num_samples}"
    )

    if len(existing) >= args.num_samples:
        print()
        print(
            "Dataset đã đạt mục tiêu."
        )
        print(
            f"File: {args.output}"
        )
        return

    # --------------------------------------------------------------
    # Tasks
    # --------------------------------------------------------------

    remaining = (
        args.num_samples
        - len(existing)
    )

    tasks = make_tasks(
        source_pool=source_pool,
        count=remaining * 3,
        seed=args.seed,
    )

    # Không đưa câu đã tồn tại vào task.
    filtered_tasks = []

    task_seen = set()

    for item in tasks:

        q = item["question"]

        key = normalize_key(q)

        if not key:
            continue

        if key in seen_questions:
            continue

        if key in task_seen:
            continue

        task_seen.add(key)
        filtered_tasks.append(item)

    tasks = filtered_tasks

    print(
        f"Cần tạo thêm   : {remaining}"
    )

    print(
        f"Task khả dụng  : {len(tasks)}"
    )

    if not tasks:
        raise RuntimeError(
            "Không còn task mới để tạo."
        )

    # --------------------------------------------------------------
    # Load teacher
    # --------------------------------------------------------------

    model = None

    try:
        model, tokenizer, input_device = load_teacher(
            args.model_name
        )

        # ----------------------------------------------------------
        # Generation
        # ----------------------------------------------------------

        print()
        print("=" * 70)
        print("Generating")
        print("=" * 70)

        kept = len(existing)
        rejected = 0
        generated = 0

        last_save = time.time()

        seen_answers = [str(row.get("assistant", "")) for row in existing]
        for index, task in enumerate(
            tasks,
            1,
        ):

            question = task["question"]

            if kept >= args.num_samples:
                break

            print()
            print("-" * 70)
            print(
                f"[{kept}/{args.num_samples}] "
                f"task {index}/{len(tasks)}"
            )
            print(
                "USER:",
                question,
            )

            try:

                answer, raw = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    input_device=input_device,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                )

            except torch.cuda.OutOfMemoryError as exc:

                rejected += 1

                print(
                    "[OOM] Bỏ mẫu này."
                )

                print(
                    repr(exc)
                )

                # Dọn cache.
                gc.collect()

                torch.cuda.empty_cache()

                continue

            except Exception as exc:

                rejected += 1

                print(
                    "[ERROR]",
                    type(exc).__name__,
                    repr(exc),
                )

                if args.debug:
                    traceback.print_exc()

                gc.collect()

                torch.cuda.empty_cache()

                continue

            generated += 1

            # ------------------------------------------------------
            # Debug raw output
            # ------------------------------------------------------

            if args.debug:
                print()
                print("RAW:")
                print(raw[:4000])

            # ------------------------------------------------------
            # Quality filter
            # ------------------------------------------------------

            bad, reason = looks_bad_answer(
                answer=answer,
                min_chars=args.min_answer_chars,
                max_chars=args.max_answer_chars,
            )

            if bad:

                rejected += 1

                print(
                    "STATUS: BỎ -",
                    reason,
                )

                continue

            if looks_self_contradictory(answer):
                rejected += 1
                print("STATUS: BỎ - có dấu hiệu tự mâu thuẫn (heuristic)")
                continue

            if semantic_duplicate(answer, seen_answers):
                rejected += 1
                print("STATUS: BỎ - câu trả lời gần trùng về từ vựng")
                continue

            # ------------------------------------------------------
            # Duplicate question
            # ------------------------------------------------------

            question_key = normalize_key(
                question
            )

            if question_key in seen_questions:

                rejected += 1

                print(
                    "STATUS: BỎ - duplicate"
                )

                continue

            # ------------------------------------------------------
            # Save
            # ------------------------------------------------------

            row = {
                "user": normalize_space(question),
                "assistant": normalize_space(answer),
                "category": task.get("category", "general"),
            }

            existing.append(row)

            seen_questions.add(
                question_key
            )
            seen_answers.append(row["assistant"])

            kept += 1

            print()
            print("ASSISTANT:")
            print(answer)

            print()
            print(
                f"STATUS: GIỮ "
                f"({kept}/{args.num_samples})"
            )

            # ------------------------------------------------------
            # Autosave
            # ------------------------------------------------------

            now = time.time()

            if (
                now - last_save >= 30
                or kept % 10 == 0
            ):

                save_dataset(
                    args.output,
                    existing,
                )

                last_save = now

                print(
                    f"[SAVE] {args.output} "
                    f"-> {len(existing)} mẫu"
                )

        # ----------------------------------------------------------
        # Final save
        # ----------------------------------------------------------

        save_dataset(
            args.output,
            existing,
        )
        evaluate_dataset(args.output)

        print()
        print("=" * 70)
        print("DONE")
        print("=" * 70)

        print(
            f"Tổng mẫu      : {len(existing)}"
        )

        print(
            f"Mẫu sinh      : {generated}"
        )

        print(
            f"Bỏ/lỗi        : {rejected}"
        )

        print(
            f"File          : {args.output}"
        )

    finally:

        if model is not None:

            try:
                del model
            except Exception:
                pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================
# ENTRY
# ======================================================================

if __name__ == "__main__":
    main()
