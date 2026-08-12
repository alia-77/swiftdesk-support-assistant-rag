import os
import time
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

client = genai.Client(api_key=API_KEY)

INPUT_FILE = Path("data/support_conversations.csv")
CHROMA_DIRECTORY = Path("chroma_db")

COLLECTION_NAME = "support_conversations"
EMBEDDING_MODEL = "gemini-embedding-2"

# Keep this comfortably below the API limit.
BATCH_SIZE = 50


def prepare_document(customer_issue):
    return f"title: IT support ticket | text: {customer_issue}"


def generate_embeddings(texts):
    contents = [
        types.Content(
            parts=[
                types.Part.from_text(text=text)
            ]
        )
        for text in texts
    ]

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
    )

    return [embedding.values for embedding in response.embeddings]


def main():
    print("Loading support conversations...")

    dataframe = pd.read_csv(INPUT_FILE)

    print(f"Number of documents: {len(dataframe)}")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    existing_ids = set(
        collection.get()["ids"]
    )

    print(f"Documents already in Chroma: {len(existing_ids)}")

    pending_rows = []

    for index, row in dataframe.iterrows():
        document_id = str(index)

        if document_id in existing_ids:
            continue

        customer_issue = str(row["customer_issue"])
        reference_reply = str(row["reference_reply"])

        pending_rows.append(
            {
                "id": document_id,
                "customer_issue": customer_issue,
                "reference_reply": reference_reply,
                "document": prepare_document(customer_issue),
            }
        )

    print(f"Documents remaining to process: {len(pending_rows)}")

    if not pending_rows:
        print("\nAll documents are already stored in Chroma.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Documents stored: {collection.count()}")
        return

    for start in range(0, len(pending_rows), BATCH_SIZE):
        batch = pending_rows[start:start + BATCH_SIZE]

        batch_number = start // BATCH_SIZE + 1
        total_batches = (
            len(pending_rows) + BATCH_SIZE - 1
        ) // BATCH_SIZE

        print(
            f"\nProcessing batch {batch_number}/{total_batches} "
            f"({len(batch)} documents)..."
        )

        texts = [
            item["document"]
            for item in batch
        ]

        # Retry the batch if a temporary quota/rate-limit error occurs.
        while True:
            try:
                embeddings = generate_embeddings(texts)
                break

            except Exception as error:
                if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
                    print(
                        "Embedding quota temporarily exhausted. "
                        "Waiting 60 seconds before retrying..."
                    )
                    time.sleep(60)
                else:
                    raise

        collection.upsert(
            ids=[
                item["id"]
                for item in batch
            ],
            embeddings=embeddings,
            documents=[
                item["customer_issue"]
                for item in batch
            ],
            metadatas=[
                {
                    "reference_reply": item["reference_reply"]
                }
                for item in batch
            ],
        )

        processed = min(
            start + BATCH_SIZE,
            len(pending_rows)
        )

        print(
            f"Batch completed. "
            f"Processed {processed}/{len(pending_rows)} remaining documents."
        )

        # Small pause between batches.
        time.sleep(2)

    print("\nChroma ingestion completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    main()

