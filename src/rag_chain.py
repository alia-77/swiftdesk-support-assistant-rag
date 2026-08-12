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
GENERATION_MODEL = "gemini-3.6-flash"


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


def build_rag_prompt(customer_issue, rag_context):
    return f"""
You are an IT support assistant responsible for drafting
professional customer support responses.

Use the retrieved support examples below as guidance.

Important instructions:
- Base the response primarily on the customer's current issue.
- Use retrieved examples only when they are relevant.
- Do not invent company policies, services, contact information,
  or technical details that are not supported by the provided context.
- Do not copy an example blindly if it does not match the customer's issue.
- Write a short, clear, polite, professional support response.
- Do not expose your reasoning process.
- The response is a draft and must be reviewed by a human support agent
  before being sent to the customer.

Retrieved support examples:

{rag_context}

Current customer issue:

{customer_issue}

Draft support reply:
"""


def generate_rag_response(customer_issue, k=3):
    rag_context = build_rag_context(
        customer_issue,
        k=k,
    )

    client = genai.Client(api_key=API_KEY)

    prompt = build_rag_prompt(
        customer_issue,
        rag_context,
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return {
        "customer_issue": customer_issue,
        "retrieved_context": rag_context,
        "draft_response": response.text,
    }


if __name__ == "__main__":
    print("rag_chain.py loaded successfully.")
    