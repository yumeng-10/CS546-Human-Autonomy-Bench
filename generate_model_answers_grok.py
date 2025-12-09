import argparse
import csv
import json
import time
from typing import Dict, Any, List
import os
from openai import OpenAI
import google.generativeai as genai
# 假设你已经在系统里设置了环境变量 OPENAI_API_KEY
# 不要在代码里硬编码 key
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
    client: OpenAI,
    model: str,
    question: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """
    调用一次模型，给单个 question 生成一个正常 assistant 答案。
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful assistant that strictly follows "
                            "the user's JSON output schema."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(
                f"[WARN] Error on attempt {attempt + 1} for question='{question[:40]}...': {e}"
            )
            time.sleep(1.5)

    raise RuntimeError("Failed to generate answer after retries.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model answers for each question."
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
        default="./results/model_answers/out_grok.jsonl",
        help="Path to output .jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="x-ai/grok-3",
        help="Model name to use for generating answers.",
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

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for q_idx, question in enumerate(questions):
            try:
                answer = generate_answer_for_question(
                    client=OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key="your_key_here",
                    ),
                    model=args.model,
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
                },
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (q_idx + 1) % 10 == 0:
                print(f"Generated answers for {q_idx + 1}/{len(questions)} questions")

    print(f"Done. Saved model answers to {args.output_jsonl}")


if __name__ == "__main__":
    main()
