import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import errors
from pydantic import BaseModel

from src.prompts import (
    few_shot_prompt,
    rag_prompt,
    reasoned_prompt,
    zero_shot_prompt,
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

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        return response.text

    except errors.ClientError as error:
        if error.code == 429:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini API quota has been exhausted. "
                    "Please try again later."
                ),
            )

        raise HTTPException(
            status_code=502,
            detail="Gemini API request failed.",
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate a response from Gemini.",
        )


@app.post("/generate", response_model=GenerateResponse)
def generate_response(request: GenerateRequest):
    customer_issue = request.customer_issue.strip()

    if not customer_issue:
        raise HTTPException(
            status_code=400,
            detail="Customer issue cannot be empty.",
        )

    if request.num_examples < 1 or request.num_examples > 10:
        raise HTTPException(
            status_code=400,
            detail="num_examples must be between 1 and 10.",
        )

    retrieved_sources = []

    if request.rag_enabled:
        try:
            retrieved_sources = retrieve_similar_tickets(
                customer_issue,
                k=request.num_examples,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="RAG retrieval failed.",
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

    draft_reply = generate_with_gemini(prompt)

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

