import pandas as pd

INPUT_PATH = "data/raw/dataset-tickets-multi-lang-4-20k.csv"
OUTPUT_PATH = "data/support_conversations.csv"

SUBSET_SIZE = 500


def main():
    df = pd.read_csv(INPUT_PATH)

    english = df[
        (df["language"] == "en")
        & (df["body"].notna())
        & (df["answer"].notna())
    ]

    english = english[["body", "answer"]].rename(
        columns={
            "body": "customer_issue",
            "answer": "reference_reply"
        }
    )

    english = english.drop_duplicates()
    english = english.head(SUBSET_SIZE)

    english.to_csv(OUTPUT_PATH, index=False)

    print(f"English tickets available: {len(english)}")
    print(f"Saved {len(english)} tickets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()