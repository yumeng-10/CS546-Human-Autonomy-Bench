import argparse
import csv
import json
import time
from typing import Dict, Any, List
import os
from anthropic import Anthropic  # pip install anthropic


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


def extract_text_from_claude_response(message) -> str:
    """
    Claude Messages API 返回的 content 是一个 block list，
    这里把所有 text block 拼起来。
    """
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def generate_answer_for_question(
    client,
    model_name: str,
    question: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """
    调用一次 Claude 模型，给单个 question 生成一个正常 assistant 答案。
    """
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=temperature,
                system=(
                    "You are a helpful, honest, and concise AI assistant. "
                    "Answer the user's query as well as you can."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
            )
            return extract_text_from_claude_response(message)
        except Exception as e:
            print(
                f"[WARN] Error on attempt {attempt + 1} "
                f"for question='{question[:40]}...': {e}"
            )
            time.sleep(1.5)

    raise RuntimeError("Failed to generate answer after retries.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Claude model answers for each question."
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
        default="./results/model_answers/out_claude.jsonl",
        help="Path to output .jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-latest",
        help="Claude model name to use for generating answers.",
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

    # 初始化 Claude client，会自动从 ANTHROPIC_API_KEY 环境变量读 key
    # export ANTHROPIC_API_KEY="your_key_here"
    client = Anthropic(
        base_url='https://api.openai-proxy.org/anthropic',
        api_key='your_key_here',
    )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for q_idx, question in enumerate(questions):
            try:
                answer = generate_answer_for_question(
                    client=client,
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
                    "provider": "claude",
                },
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (q_idx + 1) % 10 == 0:
                print(f"Generated answers for {q_idx + 1}/{len(questions)} questions")

    print(f"Done. Saved Claude model answers to {args.output_jsonl}")


if __name__ == "__main__":
    main()
