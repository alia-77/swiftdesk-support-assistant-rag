import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

client = genai.Client(api_key=API_KEY)

INPUT_FILE = Path("data/test_subset.json")
OUTPUT_FILE = Path("outputs/baseline_outputs.json")


def generate_response(prompt):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


def zero_shot_prompt(customer_issue):
    return f"""
You are an IT support assistant.

Write a short, clear, polite support reply to the customer.

Customer issue:
{customer_issue}

Reply:
"""


def few_shot_prompt(customer_issue, examples):
    examples_text = "\n\n".join(
        [
            f"Customer issue:\n{example['customer_issue']}\n"
            f"Support reply:\n{example['reference_reply']}"
            for example in examples
        ]
    )

    return f"""
You are an IT support assistant.

Use the examples below as guidance for writing a short,
clear, polite support reply.

Examples:

{examples_text}

Now write a support reply for this customer issue:

{customer_issue}

Reply:
"""


def reasoned_prompt(customer_issue):
    return f"""
You are an IT support assistant.

Analyze the customer's issue carefully and identify the
main problem before drafting the response.

Then write a short, clear, polite support reply.
Do not expose your internal reasoning.

Customer issue:
{customer_issue}

Reply:
"""


def load_existing_results():
    if not OUTPUT_FILE.exists():
        return []

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        test_examples = json.load(file)

    results = load_existing_results()

    completed_task_ids = {
        result["task_id"]
        for result in results
    }

    print(f"Already completed: {len(completed_task_ids)} task(s)")

    for example in test_examples:
        task_id = example["task_id"]

        if task_id in completed_task_ids:
            print(f"Skipping task {task_id} - already completed.")
            continue

        customer_issue = example["customer_issue"]

        print(f"Processing task {task_id}...")

        zero_shot = generate_response(
            zero_shot_prompt(customer_issue)
        )
        time.sleep(13)

        few_shot_examples = test_examples[:2]

        few_shot = generate_response(
            few_shot_prompt(customer_issue, few_shot_examples)
        )
        time.sleep(13)

        reasoned = generate_response(
            reasoned_prompt(customer_issue)
        )
        time.sleep(13)

        results.append(
            {
                "task_id": task_id,
                "customer_issue": customer_issue,
                "reference_reply": example["reference_reply"],
                "zero_shot": zero_shot,
                "few_shot": few_shot,
                "reasoned": reasoned,
            }
        )

        save_results(results)

        print(f"Task {task_id} completed and saved.")

    print(f"\nSaved results to {OUTPUT_FILE}")
    print(f"Number of completed test examples: {len(results)}")


if __name__ == "__main__":
    main()
