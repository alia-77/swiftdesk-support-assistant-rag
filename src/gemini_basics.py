import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from prompts import rag_prompt
from rag_chain import retrieve_similar_tickets


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


def generate_rag_response(customer_issue):
    retrieved_sources = retrieve_similar_tickets(
        customer_issue,
        k=3,
    )

    prompt = rag_prompt(
        customer_issue,
        retrieved_sources,
    )

    response = generate_response(prompt)

    return response, retrieved_sources


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

        print(f"\nChecking task {task_id}...")

        if "zero_shot" in result:
            print("  Zero-shot already exists. Skipping.")
        else:
            print("  Zero-shot is missing. Skipping baseline generation.")

        if "few_shot" in result:
            print("  Few-shot already exists. Skipping.")
        else:
            print("  Few-shot is missing. Skipping baseline generation.")

        if "reasoned" in result:
            print("  Reasoned already exists. Skipping.")
        else:
            print("  Reasoned is missing. Skipping baseline generation.")

        if "rag" in result:
            print("  RAG already exists. Skipping.")
            continue

        print("  Generating RAG response...")

        rag_response, retrieved_sources = generate_rag_response(
            customer_issue
        )

        result["rag"] = rag_response
        result["retrieved_sources"] = retrieved_sources

        save_results(
            [
                result_by_task[example["task_id"]]
                for example in test_examples
            ]
        )

        print("  RAG response saved.")

        time.sleep(13)

        print(f"Task {task_id} RAG processing completed.")

    final_results = [
        result_by_task[example["task_id"]]
        for example in test_examples
    ]

    save_results(final_results)

    completed_rag = sum(
        1
        for result in final_results
        if "rag" in result
    )

    print(f"\nSaved results to {OUTPUT_FILE}")
    print(
        f"Completed RAG examples: "
        f"{completed_rag}/{len(test_examples)}"
    )


if __name__ == "__main__":
    main()