# SPEC THEO TỪNG BATCH

Mỗi batch: 100 dòng, trừ khi ghi khác. Output ghi vào `out/<ID>.jsonl`.
Đọc `RULES.md` và `existing_topics.txt` trước. Không vượt quá 900 ký tự/mục.

---

## A1 — AI/LLM: kiến trúc và cơ chế bên trong
Output: `out/A1.jsonl` — 100 mục.
Phạm vi (chọn và phân bổ, không cần đúng thứ tự):
- multi-head attention, MQA, GQA, MLA: khác biệt và cái giá phải trả
- RoPE: áp lên Q/K, vì sao không áp lên V, ảnh hưởng khi context dài
- ALiBi, positional embedding học được, vì sao RoPE phổ biến
- pre-norm vs post-norm, LayerNorm vs RMSNorm, vì sao RMSNorm nhanh hơn
- residual connection chống vanishing gradient như thế nào
- GeLU, SiLU/Swish, SwiGLU: khác nhau ở đâu
- MoE: routing top-k, load balancing, expert collapse
- KV cache: tăng trưởng bộ nhớ, sliding window, paged attention
- causal mask: vì sao cần, lỗi nếu thiếu
- tokenizer: BPE vs WordPiece vs SentencePiece, vocab size ảnh hưởng gì
- weight tying, embedding scale, init
- FlashAttention: giảm bộ nhớ bằng cách nào
- context window mở rộng: khó khăn thực tế
- scaling law: tham số vs dữ liệu vs compute
- sinh văn bản: temperature, top-k, top-p, repetition penalty phối hợp
Chống trùng: không hỏi lại "transformer là gì", "attention là gì", "RoPE là gì".

## A2 — AI/ML: huấn luyện và tối ưu
Output: `out/A2.jsonl` — 100 mục.
Phạm vi:
- SGD, momentum, Adam, AdamW, Lion: khi nào chọn cái nào
- weight decay vs L2, decoupled weight decay
- LR schedule: warmup, cosine, step decay; LR quá cao/quá thấp biểu hiện ra sao
- gradient accumulation: khi nào cần, có đổi kết quả không
- gradient clipping: norm vs value, chọn ngưỡng
- mixed precision fp16 vs bf16, loss scaling
- DDP, FSDP, zero redundancy, all-reduce
- checkpoint: lưu gì, resume sai thì hỏng gì
- batch size ảnh hưởng convergence và bộ nhớ
- data packing, padding, padding token có bị tính loss không
- đọc loss curve: spike, plateau, dao động
- overfit/underfit: dấu hiệu và cách xử lý
- dropout, label smoothing, early stopping, data augmentation
- khởi tạo trọng số và ảnh hưởng đến huấn luyện
- reproducibility: seed, deterministic, nguyên nhân kết quả lệch
- catastrophic forgetting khi huấn luyện tiếp
- debug NaN loss: nguyên nhân thường gặp

## A3 — AI/LLM: fine-tuning, alignment, nén model
Output: `out/A3.jsonl` — 100 mục.
Phạm vi:
- SFT: dữ liệu bao nhiêu là đủ, chất lượng vs số lượng
- instruction tuning: format, tỉ lệ domain, chống học vẹt
- LoRA: rank, alpha, target module, vì sao tiết kiệm
- QLoRA, 4-bit NF4, double quantization
- PEFT khác full fine-tune ở đâu
- RLHF pipeline, reward model, PPO, DPO khác PPO thế nào
- distillation: teacher-student, soft label, temperature
- quantization: GPTQ, AWQ, GGUF, int8/int4, mất mát gì
- pruning, structured vs unstructured
- model merging, task arithmetic
- benchmark: contamination, vì sao điểm cao vẫn dở
- hallucination: nguyên nhân và cách giảm
- prompt injection, jailbreak, guardrail
- safety alignment, refusal cân bằng
- continual pretraining, domain adaptation
- serving: batch, KV cache reuse, throughput vs latency

## A4 — ML cổ điển, retrieval, agent
Output: `out/A4.jsonl` — 100 mục.
Phạm vi:
- supervised vs unsupervised vs self-supervised, ví dụ
- bias-variance, regularization, đường cong học
- decision tree, random forest, gradient boosting (XGBoost/LightGBM)
- SVM, kernel trick, kNN, ưu nhược điểm
- k-means, DBSCAN, chọn k
- PCA, t-SNE, chuẩn hóa dữ liệu trước PCA
- metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC; khi nào dùng
- confusion matrix, threshold tuning
- cross-validation, stratified, time-series split
- data leakage, ví dụ tinh vi
- imbalanced data: resampling, class weight
- feature engineering: scaling, one-hot, target encoding
- naive Bayes, logistic regression
- RAG: chunking, overlap, embedding, vector DB, rerank
- đánh giá RAG: faithfulness, context recall
- agent: tool calling, ReAct, vòng lặp, giới hạn
- so sánh fine-tune vs RAG

## B1 — C++ cơ bản và mô hình bộ nhớ
Output: `out/B1.jsonl` — 100 mục.
Phạm vi:
- kiểu dữ liệu, sizeof, tràn số, chuyển đổi có/không tường minh
- const, constexpr, consteval, const correctness
- reference vs pointer, reference hết hạn
- stack vs heap, lifetime, dangling pointer
- RAII, destructor, smart pointer (unique/shared/weak)
- move semantics, std::move, copy elision, RVO
- struct vs class, access specifier
- enum vs enum class
- static local, static member, inline variable
- nullptr vs NULL vs 0
- mảng C vs std::vector vs std::array
- std::string vs const char*, string_view
- cin/cout, getline, lỗi nhập liệu
- UB thường gặp: tràn mảng, signed overflow, đọc biến chưa khởi tạo
- sizeof/alignment cơ bản
- vòng lặp, phạm vi biến, shadowing

## B2 — C++ OOP, template, hiện đại
Output: `out/B2.jsonl` — 100 mục.
Phạm vi:
- kế thừa, virtual, vtable, overriding vs overloading
- abstract class, pure virtual, interface
- đa kế thừa, diamond, virtual inheritance
- destructor ảo, vì sao cần
- rule of 0/3/5
- copy constructor, copy assignment, self-assignment
- template function/class, specialization, variadic template
- concepts và requires (C++20)
- lambda, capture theo giá trị/tham chiếu, mutable
- std::function, function pointer, callback
- operator overloading, friend, explicit
- exception, noexcept, stack unwinding
- RTTI, dynamic_cast, typeid
- CRTP, policy-based design
- auto, decltype, structured binding
- initializer_list, uniform initialization

## B3 — C++ STL và container
Output: `out/B3.jsonl` — 100 mục.
Phạm vi:
- vector: capacity, reserve, reallocation, iterator invalidation
- deque, list, forward_list: khi nào dùng
- map vs unordered_map: thứ tự, độ phức tạp, hash
- set/multiset, custom comparator
- priority_queue: max-heap mặc định, min-heap
- stack, queue, pair, tuple
- iterator categories, begin/end, reverse
- thuật toán: sort, stable_sort, lower_bound, unique, accumulate, transform
- erase-remove idiom
- string_view, span
- hash function tự định nghĩa
- bộ nhớ đệm và locality ảnh hưởng container
- chọn container theo workload
- lỗi thường gặp: dùng iterator sau khi xóa
- emplace vs insert
- truyền container vào hàm: const ref vs value

## B4 — C++ concurrency, build, debug
Output: `out/B4.jsonl` — 100 mục.
Phạm vi:
- thread, join, detach, lifetime
- mutex, lock_guard, unique_lock, scoped_lock
- deadlock: điều kiện và cách tránh
- condition_variable, spurious wakeup
- atomic, memory_order, happens-before
- race condition ví dụ thực tế
- data race vs race condition
- async, future, promise
- thread pool cơ bản
- false sharing, cache coherence
- mutex vs atomic: chi phí
- CMake cơ bản: target, include, link
- header vs source, include guard, ODR
- quá trình biên dịch: tiền xử lý, biên dịch, liên kết
- lỗi linker thường gặp (undefined reference, multiple definition)
- sanitizer: ASan, UBSan, TSan
- gdb cơ bản: breakpoint, backtrace
- tối ưu -O2/-O3 và ảnh hưởng hành vi

## B5 — Thuật toán và cấu trúc dữ liệu
Output: `out/B5.jsonl` — 100 mục.
Phạm vi:
- độ phức tạp thời gian/bộ nhớ, amortized
- two pointers, sliding window
- prefix sum, difference array
- binary search: biến thể tìm biên, điều kiện đúng
- sắp xếp: quick/merge/heap, stability, khi nào dùng
- hash table: collision, load factor
- linked list: đảo, tìm giữa, phát hiện cycle
- stack/queue ứng dụng: ngoặc đúng, monotonic stack
- heap, top-k, merge k list
- cây: duyệt pre/in/post/level, BST thao tác
- balanced tree, AVL vs red-black (mức khái niệm)
- trie: ứng dụng
- đồ thị: biểu diễn, BFS/DFS, topo sort, union-find, Dijkstra
- DP: knapsack, LIS, edit distance, đổi tiền
- greedy: khi nào đúng, phản ví dụ
- backtracking: N-queens, subset
- bit manipulation: kiểm tra bit, lowbit, subset mask
- đệ quy và memoization
- lỗi off-by-one, tràn số trong bài đếm
Chống trùng: không hỏi lại "DFS là gì", "heap là gì", "dynamic programming là gì".

## C1 — Python cơ bản và ngôn ngữ
Output: `out/C1.jsonl` — 100 mục.
Phạm vi:
- int/float, chia lấy dư, làm tròn, float precision
- str methods, f-string, format, encode/decode
- list/dict/set/tuple thao tác và độ phức tạp
- slicing, step âm, copy nông
- unpacking, star args, keyword-only
- comprehension, điều kiện lọc, lồng nhau
- generator, yield, lazy evaluation
- iterator vs iterable
- closure, late binding trong vòng lặp
- decorator, functools.wraps, decorator có tham số
- context manager, với open
- exception: try/except/else/finally, custom exception
- LEGB, global/nonlocal
- truthiness, so sánh `is` vs `==`
- sắp xếp với key, sorted ổn định, nhiều khóa
- enumerate, zip, zip strict
- walrus operator
- mutable default argument
- hashable, vì sao list không làm key
- hiệu năng vòng lặp Python
Chống trùng: tránh "list là gì", "tuple là gì", "lambda là gì", "decorator là gì".

## C2 — Python OOP, typing, testing, tooling
Output: `out/C2.jsonl` — 100 mục.
Phạm vi:
- class, __init__, instance vs class attribute
- dataclass, field, default_factory, frozen
- __slots__ lợi ích
- property, setter, computed attribute
- staticmethod vs classmethod vs instance method
- kế thừa, super(), MRO, đa kế thừa
- ABC, abstractmethod, Protocol
- type hints: Optional, Union, list[int], Callable
- TypeVar, generic class
- mypy: lỗi thường gặp
- pytest: fixture, parametrize, assert
- unittest, mock, patch
- logging: level, handler, format
- argparse, subcommand
- pathlib, os.path, đường dẫn Windows
- venv, pip, requirements, pyproject
- packaging, entry point
- ruff/black/flake8, pre-commit
- debug bằng breakpoint/pdb
- profiling: cProfile, timeit
- code style, docstring

## C3 — Python thực dụng, dữ liệu, bất đồng bộ
Output: `out/C3.jsonl` — 100 mục.
Phạm vi:
- đọc/ghi file text, encoding utf-8, newline
- CSV đọc ghi, DictReader
- JSON load/dump, ensure_ascii, indent
- regex: nhóm, greedy, raw string
- datetime, timezone, timedelta, parse
- itertools: chain, groupby, product, combinations
- collections: Counter, defaultdict, deque, namedtuple
- NumPy: array vs list, broadcasting, axis
- pandas: DataFrame, loc/iloc, groupby, merge
- requests: timeout, status code, JSON
- async/await, event loop, asyncio.gather
- GIL, threading vs multiprocessing vs asyncio
- xử lý file lớn bằng generator
- retry/backoff, xử lý lỗi mạng
- biến môi trường, dotenv
- subprocess, shell=True rủi ro
- unit test cho hàm có IO
- lỗi runtime thường gặp: KeyError, IndexError, TypeError
- memory leak do giữ tham chiếu
- clean code: hàm nhỏ, tên rõ, tránh lồng sâu

## D1 — Toán
Output: `out/D1.jsonl` — 100 mục.
Phạm vi:
- phương trình bậc 1/2, hệ phương trình
- bất phương trình, dấu giá trị tuyệt đối
- hàm số, tập xác định, tính đơn điệu
- cấp số cộng/nhân, quy luật dãy
- logarit, mũ, đổi cơ số
- hình học phẳng: tam giác, đường tròn, diện tích
- lượng giác: giá trị đặc biệt, hệ thức
- xác suất: biến cố, độc lập, có điều kiện
- tổ hợp, chỉnh hợp, hoán vị
- thống kê: trung bình, trung vị, phương sai, độ lệch chuẩn
- đạo hàm: quy tắc, ứng dụng cực trị
- tích phân cơ bản, diện tích
- ma trận: nhân, định thức, nghịch đảo
- số phức: dạng đại số, môđun
- giới hạn hàm số
- bài toán thực tế: lãi suất, chuyển động, nồng độ
Mỗi bài tính phải có đáp án đúng, kiểm tra lại.

## D2 — Vật lý
Output: `out/D2.jsonl` — 100 mục.
Phạm vi:
- chuyển động thẳng đều/biến đổi đều, đồ thị
- ba định luật Newton, lực ma sát, mặt phẳng nghiêng
- công, công suất, động năng, thế năng, bảo toàn cơ năng
- động lượng, va chạm đàn hồi/không đàn hồi
- chuyển động tròn, lực hướng tâm
- hấp dẫn, trọng lượng vs khối lượng
- áp suất chất lỏng, lực đẩy Archimedes
- nhiệt: nóng chảy, bay hơi, truyền nhiệt
- khí lý tưởng: Boyle, Charles, phương trình trạng thái
- sóng, âm thanh, tần số, bước sóng
- ánh sáng: khúc xạ, phản xạ, thấu kính
- điện: định luật Ohm, mạch nối tiếp/song song, công suất điện
- từ trường, cảm ứng điện từ
- đơn vị đo, đổi đơn vị, sai số
- hiểu lầm thường gặp (khối lượng vs trọng lượng, vận tốc vs tốc độ)
Mỗi bài tính phải có đáp án đúng.

## D3 — Hóa học
Output: `out/D3.jsonl` — 100 mục.
Phạm vi:
- cấu tạo nguyên tử, electron lớp ngoài, đồng vị
- bảng tuần hoàn, xu hướng bán kính/độ âm điện
- liên kết ion, cộng hóa trị, kim loại, liên kết hydro
- mol, khối lượng mol, số Avogadro
- cân bằng phương trình, stoichiometry
- dung dịch, nồng độ mol, pha loãng
- pH, axit mạnh/yếu, bazơ, trung hòa
- oxi hóa – khử, xác định số oxi hóa
- phản ứng: hóa hợp, phân hủy, thế, trao đổi
- nhiệt phản ứng, tỏa/thu nhiệt, enthalpy
- tốc độ phản ứng, xúc tác, cân bằng Le Chatelier
- chất khí, thể tích mol ở đktc
- hữu cơ: ankan, anken, ancol, axit axetic, este
- công thức hóa học thường gặp
- an toàn phòng thí nghiệm, cách pha hóa chất
Mỗi bài tính phải có đáp án đúng.

## E1 — Hội thoại nhiều lượt: đời sống và học tập
Output: `out/E1.jsonl` — 100 mục.
Định dạng đặc biệt: `user` chứa 2–3 lượt trước, mỗi lượt một dòng, theo mẫu:
`Người dùng: ...\nTrợ lý: ...\nNgười dùng: ...`
`assistant` là câu trả lời cho lượt Người dùng cuối cùng.
Tổng độ dài vẫn ≤ 900 ký tự, nên lượt trước phải ngắn.
Phạm vi:
- chào hỏi, giới thiệu, hỏi lại cho rõ
- lập kế hoạch học tập, chia thời gian
- giải thích rồi hỏi tiếp, đào sâu
- sửa hiểu lầm của người dùng
- hỏi preference rồi đề xuất
- tóm tắt thống nhất, chốt việc cần làm
- xin ví dụ, xin ví dụ khác
- so sánh hai lựa chọn qua nhiều lượt
- phản hồi "chưa hiểu", yêu cầu đơn giản hơn
- chuyển chủ đề giữa cuộc trò chuyện
Mỗi mục phải thể hiện được ngữ cảnh phụ thuộc lượt trước (đại từ, "cái đó", "vậy còn...").

## E2 — Hội thoại nhiều lượt: gỡ lỗi kỹ thuật
Output: `out/E2.jsonl` — 100 mục.
Cùng định dạng nhiều lượt như E1.
Phạm vi:
- lỗi Python: ImportError, ModuleNotFoundError, KeyError, TypeError, IndentationError
- lỗi C++: undefined reference, segmentation fault, lỗi biên dịch template
- môi trường: pip, venv, PATH trên Windows
- git: merge conflict, detached HEAD, commit nhầm
- dữ liệu: encoding, CSV lỗi, JSON hỏng
- hiệu năng: chương trình chậm, memory tăng
- hỏi thêm thông tin trước khi kết luận
- đưa giả thuyết, người dùng xác nhận, chốt cách sửa
- người dùng làm theo nhưng vẫn lỗi, điều chỉnh
Mỗi mục nên có: triệu chứng → hỏi/kiểm tra → nguyên nhân → cách sửa ngắn.

## E3 — Hội thoại nhiều lượt: gia sư và tư vấn
Output: `out/E3.jsonl` — 100 mục.
Cùng định dạng nhiều lượt như E1.
Phạm vi:
- học sinh hỏi khái niệm, xin ví dụ, hỏi trường hợp biên
- giải bài rồi hỏi "nếu đổi số thì sao"
- tư vấn chọn ngôn ngữ/công cụ theo mục tiêu
- tư vấn lộ trình học AI, C++, Python
- đánh đổi giữa tốc độ và chất lượng
- người dùng do dự, trợ lý đưa tiêu chí quyết định
- sửa một hiểu lầm về thuật toán
- hướng dẫn từng bước, người dùng hỏi bước tiếp
- xin tóm tắt sau khi trao đổi dài

## F1 — Suy luận định lượng và logic
Output: `out/F1.jsonl` — 100 mục.
Phạm vi:
- bài toán công việc chung (hai người làm chung)
- chuyển động, gặp nhau, đuổi kịp
- tuổi, tổng–hiệu, tỉ lệ
- phần trăm tăng/giảm, lãi kép
- pha trộn nồng độ, quy tắc đường chéo
- dãy số quy luật, tìm số tiếp theo
- lập lịch, xếp chỗ, thứ tự
- suy luận đúng/sai với điều kiện
- Venn 2–3 tập
- xác suất đơn giản, không thay thế
- đổi đơn vị, ước lượng
- đọc bảng số liệu ngắn (đưa số ngay trong câu hỏi)
- bài toán cân, đong nước, qua sông
Mỗi câu phải có đủ dữ kiện và đáp án đúng; trình bày 2–3 bước lập luận ngắn.

## F2 — Giải thích, phân tích vấn đề
Output: `out/F2.jsonl` — 100 mục.
Phạm vi:
- chẩn đoán nguyên nhân từ triệu chứng (hệ thống, code, dữ liệu)
- phân tích đánh đổi giữa hai phương án kỹ thuật
- tìm lỗ hổng trong một lập luận ngắn do người dùng đưa ra
- giải thích số liệu: loss tăng, accuracy giảm, latency tăng
- chọn cấu trúc dữ liệu cho một tình huống cụ thể
- ước lượng độ phức tạp từ đoạn code ngắn
- lập kế hoạch từng bước cho một việc cụ thể
- giải thích cho người mới bằng phép so sánh
- phát hiện giả định sai trong yêu cầu
- đề xuất cách kiểm chứng một giả thuyết
Mỗi mục: nêu tình huống cụ thể, yêu cầu phân tích; đáp án đi thẳng vào nguyên nhân/tiêu chí, không liệt kê chung chung.
