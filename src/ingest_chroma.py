import os
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

client = genai.Client(api_key=API_KEY)

INPUT_FILE = Path("data/support_conversations.csv")
CHROMA_DIRECTORY = Path("chroma_db")

COLLECTION_NAME = "support_conversations"
EMBEDDING_MODEL = "gemini-embedding-2"


def prepare_document(customer_issue):
    return f"title: IT support ticket | text: {customer_issue}"


def generate_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


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

    print("Creating embeddings and storing documents...")

    for index, row in dataframe.iterrows():
        customer_issue = str(row["customer_issue"])
        reference_reply = str(row["reference_reply"])

        document = prepare_document(customer_issue)
        embedding = generate_embedding(document)

        collection.upsert(
            ids=[str(index)],
            embeddings=[embedding],
            documents=[customer_issue],
            metadatas=[
                {
                    "reference_reply": reference_reply
                }
            ],
        )

        print(f"Processed {index + 1}/{len(dataframe)}")

    print("\nChroma ingestion completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    main()

