import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/support_conversations.csv")
OUTPUT_FILE = Path("data/test_subset.json")


def main():
    df = pd.read_csv(INPUT_FILE)

    test_df = df.head(10)

    test_examples = []

    for task_id, row in enumerate(test_df.itertuples(index=False), start=1):
        test_examples.append(
            {
                "task_id": task_id,
                "customer_issue": row.customer_issue,
                "reference_reply": row.reference_reply,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(test_examples, file, indent=2, ensure_ascii=False)

    print(f"Created {OUTPUT_FILE}")
    print(f"Number of examples: {len(test_examples)}")


if __name__ == "__main__":
    main()