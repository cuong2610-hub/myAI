import os
import subprocess

PROMPTS = [
    "Thủ đô của Việt Nam là gì?",
    "Giải thích ngắn gọn C++ là gì.",
    "Python dùng để làm gì?",
    "2 + 2 bằng bao nhiêu?",
    "Định luật bảo toàn năng lượng phát biểu thế nào?",
    "Transformer là gì?",
    "Viết C++ tính tổng từ 1 đến n.",
    "What is machine learning?",
    "Explain what a pointer is in C++.",
    "What does a dictionary do in Python?",
]

print("Bộ prompt kiểm thử AI5/AI7. Dùng chat.py để thử trực tiếp.")
for i, p in enumerate(PROMPTS, 1):
    print(f"{i:2d}. {p}")
