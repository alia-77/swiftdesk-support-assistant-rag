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
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        test_examples = json.load(file)

    results = load_existing_results()

    result_by_task = {
        result["task_id"]: result
        for result in results
    }

    print(
        f"Existing results found: "
        f"{len(result_by_task)} task(s)"
    )

    for example in test_examples:
        task_id = example["task_id"]
        customer_issue = example["customer_issue"]

        if task_id not in result_by_task:
            result_by_task[task_id] = {
                "task_id": task_id,
                "customer_issue": customer_issue,
                "reference_reply": example["reference_reply"],
            }

        result = result_by_task[task_id]

        print(f"\nProcessing task {task_id}...")

        if "zero_shot" not in result:
            print("  Generating zero-shot response...")

            result["zero_shot"] = generate_response(
                zero_shot_prompt(customer_issue)
            )

            save_results(list(result_by_task.values()))

            print("  Zero-shot response saved.")
            time.sleep(13)
        else:
            print("  Zero-shot already exists. Skipping.")

        if "few_shot" not in result:
            print("  Generating few-shot response...")

            few_shot_examples = test_examples[:2]

            result["few_shot"] = generate_response(
                few_shot_prompt(
                    customer_issue,
                    few_shot_examples,
                )
            )

            save_results(list(result_by_task.values()))

            print("  Few-shot response saved.")
            time.sleep(13)
        else:
            print("  Few-shot already exists. Skipping.")

        if "reasoned" not in result:
            print("  Generating reasoned response...")

            result["reasoned"] = generate_response(
                reasoned_prompt(customer_issue)
            )

            save_results(list(result_by_task.values()))

            print("  Reasoned response saved.")
            time.sleep(13)
        else:
            print("  Reasoned response already exists. Skipping.")

        print(f"Task {task_id} completed.")

    final_results = [
        result_by_task[example["task_id"]]
        for example in test_examples
    ]

    save_results(final_results)

    completed_tasks = sum(
        1
        for result in final_results
        if all(
            key in result
            for key in [
                "zero_shot",
                "few_shot",
                "reasoned",
            ]
        )
    )

    print(f"\nSaved results to {OUTPUT_FILE}")
    print(
        f"Completed test examples: "
        f"{completed_tasks}/{len(test_examples)}"
    )


if __name__ == "__main__":
    main()

