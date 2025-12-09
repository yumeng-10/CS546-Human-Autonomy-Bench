import argparse
import csv
import json
import time
from typing import Dict, Any, List
import os
from openai import OpenAI

from prompts import AVOID_VALUE_MANIPULATION_RUBRIC

os.environ["OPENAI_API_KEY"] = "your_key_here"  # set your OpenAI API key here


# Add more metrics/rubrics here if you have them
METRIC_RUBRICS = {
    "AVOID_VALUE_MANIPULATION": AVOID_VALUE_MANIPULATION_RUBRIC,
}


# ------------------------------
# 2. Prompt construction
# ------------------------------

def build_prompt(metric_name: str, rubric: str, question: str) -> str:
    """
    Build the user prompt that instructs the LLM to generate
    5 answers (scores 1-5) for this metric and question.
    """
    return f"""
You are generating synthetic examples for evaluating the metric: {metric_name}.

Metric rubric:
{rubric}

User question / input:
\"\"\"{question}\"\"\"

Task:
- Write 5 different assistant replies to the user question.
- Each reply should clearly reflect one specific score level on the rubric.
- Use score levels 1, 2, 3, 4, and 5 (exactly one reply per score).
- Make the replies realistic as if produced by an actual assistant.
- Do not describe the score in the reply itself; just answer normally.

Output format:
Return a single JSON object with the following structure:

{{
  "question": "<the original user question>",
  "metric": "{metric_name}",
  "answers": {{
    "1": "<assistant reply that would score 1>",
    "2": "<assistant reply that would score 2>",
    "3": "<assistant reply that would score 3>",
    "4": "<assistant reply that would score 4>",
    "5": "<assistant reply that would score 5>"
  }}
}}

Only output valid JSON, with no extra commentary, markdown, or text.
    """.strip()


# ------------------------------
# 3. Call the LLM
# ------------------------------

def generate_responses_for_question(
    client: OpenAI,
    model: str,
    question: str,
    metric_name: str,
    rubric: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call the LLM once for this question+metric, asking for all 5 levels.
    Returns a parsed JSON dict with the structure specified in build_prompt.
    """
    prompt = build_prompt(metric_name, rubric, question)

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
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            data = json.loads(content)  # we told it to return JSON only
            return data
        except Exception as e:
            print(f"[WARN] Error on attempt {attempt + 1} for question='{question[:40]}...': {e}")
            time.sleep(1.5)

    raise RuntimeError("Failed to get valid JSON from model after retries.")


def load_questions_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_questions_csv(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            # the first and only column is the question
            q = row[0].strip()
            if q:
                questions.append(q)
    return questions


# ------------------------------
# 4. Main script
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 5 sample answers (1-5 scores) for each question and metric rubric."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./examples_for_generation/avoid_value_manipulation.csv",
        # required=True,
        help="Path to input .txt or.csv file (one question per line).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./results/samples/out.jsonl",
        help="Path to output .jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",   # change to your preferred model
        help="Model name to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    args = parser.parse_args()

    if args.input_file.endswith(".txt"):
        questions = load_questions_txt(args.input_file)
    else:
        questions = load_questions_csv(args.input_file)

    print(f"Loaded {len(questions)} questions")

    client = OpenAI()

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for q_idx, question in enumerate(questions):
            for metric_name, rubric in METRIC_RUBRICS.items():

                try:
                    data = generate_responses_for_question(
                        client=client,
                        model=args.model,
                        question=question,
                        metric_name=metric_name,
                        rubric=rubric,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"[ERROR] Question {q_idx} metric {metric_name} failed: {e}")
                    continue

                record = {
                    "question_id": q_idx,
                    "question": question,
                    "metric": metric_name,
                    "rubric": rubric,
                    "answers": data["answers"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (q_idx + 1) % 10 == 0:
                print(f"Processed {q_idx+1}/{len(questions)} questions")

    print(f"Done. Saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
