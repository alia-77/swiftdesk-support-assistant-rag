import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

from src.prompts import (
    zero_shot_prompt,
    few_shot_prompt,
    reasoned_prompt,
    rag_prompt,
)
from src.rag_chain import retrieve_similar_tickets


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

if not API_KEY and not MOCK_MODE:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


app = FastAPI(
    title="SwiftDesk IT Support Assistant",
    description="RAG-based IT support response drafting assistant",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    customer_issue: str
    prompt_style: str = "zero-shot"
    rag_enabled: bool = True
    num_examples: int = 3


class GenerateResponse(BaseModel):
    draft_reply: str
    retrieved_sources: list


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
    }


def generate_with_gemini(prompt):
    if MOCK_MODE:
        return "This is a mock support response for testing."

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


@app.post("/generate", response_model=GenerateResponse)
def generate_response(request: GenerateRequest):
    customer_issue = request.customer_issue.strip()

    if not customer_issue:
        raise HTTPException(
            status_code=400,
            detail="Customer issue cannot be empty.",
        )

    retrieved_sources = []

    if request.rag_enabled:
        try:
            retrieved_sources = retrieve_similar_tickets(
                customer_issue,
                k=request.num_examples,
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"RAG retrieval failed: {str(error)}",
            )

    if request.prompt_style == "zero-shot":
        prompt = zero_shot_prompt(customer_issue)

    elif request.prompt_style == "few-shot":
        prompt = few_shot_prompt(
            customer_issue,
            retrieved_sources,
        )

    elif request.prompt_style == "reasoned":
        prompt = reasoned_prompt(customer_issue)

    elif request.prompt_style == "rag":
        prompt = rag_prompt(
            customer_issue,
            retrieved_sources,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid prompt style. "
                "Use zero-shot, few-shot, reasoned, or rag."
            ),
        )

    try:
        draft_reply = generate_with_gemini(prompt)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini generation failed: {str(error)}",
        )

    return GenerateResponse(
        draft_reply=draft_reply,
        retrieved_sources=retrieved_sources,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

