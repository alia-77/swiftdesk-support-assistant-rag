import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")


CHROMA_DIRECTORY = Path("chroma_db")
COLLECTION_NAME = "support_conversations"
EMBEDDING_MODEL = "gemini-embedding-2"


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )

            embeddings.append(response.embeddings[0].values)

        return embeddings

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        return response.embeddings[0].values


def get_vectorstore():
    embeddings = GeminiEmbeddings(API_KEY)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=embeddings,
    )


def retrieve_similar_tickets(customer_issue, k=3):
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        customer_issue,
        k=k,
    )

    retrieved_examples = []

    for document in results:
        retrieved_examples.append(
            {
                "customer_issue": document.page_content,
                "reference_reply": document.metadata.get(
                    "reference_reply", ""
                ),
            }
        )

    return retrieved_examples


def build_rag_context(customer_issue, k=3):
    retrieved_examples = retrieve_similar_tickets(
        customer_issue,
        k=k,
    )

    context_parts = []

    for example in retrieved_examples:
        context_parts.append(
            f"Previous customer issue:\n"
            f"{example['customer_issue']}\n\n"
            f"Approved support reply:\n"
            f"{example['reference_reply']}"
        )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    print("rag_chain.py loaded successfully.")

