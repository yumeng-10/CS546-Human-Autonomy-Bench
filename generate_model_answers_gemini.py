import argparse
import csv
import json
import time
from typing import Dict, Any, List
import os
import google.generativeai as genai

# 不要在代码里硬编码 key，建议在环境里设置：
# export GEMINI_API_KEY="your_key_here"


def load_questions_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_questions_csv(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            q = row[0].strip()
            if q:
                questions.append(q)
    return questions


def generate_answer_for_question(
    model_name: str,
    question: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """
    调用一次 Gemini 模型，给单个 question 生成一个正常 assistant 答案。
    这里每次调用都新建一个 GenerativeModel，方便看懂；如果你追求性能，
    可以在 main 里建一个 model 对象传进来复用。
    """
    # 带 system instruction 的模型
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=(
            "You are a helpful, honest, and concise AI assistant. "
            "Answer the user's query as well as you can."
        ),
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                question,
                generation_config={
                    "temperature": temperature,
                },
            )
            # google-generativeai 默认把文本拼到 response.text 里
            return response.text
        except Exception as e:
            print(
                f"[WARN] Error on attempt {attempt + 1} "
                f"for question='{question[:40]}...': {e}"
            )
            time.sleep(1.5)

    raise RuntimeError("Failed to generate answer after retries.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini model answers for each question."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./examples_for_generation/avoid_value_manipulation.csv",
        help="Path to input .txt or .csv file (one question per line).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./results/model_answers/out_gemini.jsonl",
        help="Path to output .jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",  # 或 gemini-1.5-pro / 2.0 等
        help="Gemini model name to use for generating answers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    args = parser.parse_args()

    # 读取问题列表（兼容 txt / csv）
    if args.input_file.endswith(".txt"):
        questions = load_questions_txt(args.input_file)
    else:
        questions = load_questions_csv(args.input_file)

    print(f"Loaded {len(questions)} questions")

    # ---- Gemini SDK 配置 ----
    # 标准方式：用官方 endpoint
   

    # 如果你必须走自己的 proxy，可以改成类似：
    genai.configure(
         api_key="your_key_here",
         transport="rest",
         client_options={"api_endpoint": "https://api.openai-proxy.org/google"},
     )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for q_idx, question in enumerate(questions):
            try:
                answer = generate_answer_for_question(
                    model_name=args.model,
                    question=question,
                    temperature=args.temperature,
                )
            except Exception as e:
                print(f"[ERROR] Question {q_idx} failed: {e}")
                continue

            record: Dict[str, Any] = {
                "question_id": q_idx,
                "question": question,
                "answer": answer,
                "meta": {
                    "model": args.model,
                    "temperature": args.temperature,
                    "provider": "gemini",
                },
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (q_idx + 1) % 10 == 0:
                print(f"Generated answers for {q_idx + 1}/{len(questions)} questions")

    print(f"Done. Saved Gemini model answers to {args.output_jsonl}")


if __name__ == "__main__":
    main()
