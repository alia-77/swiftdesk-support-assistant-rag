# SwiftDesk IT Support Assistant

A RAG-based IT support assistant for drafting customer support responses using relevant historical support conversations as context.

The system retrieves similar support tickets from a ChromaDB vector store and uses their approved replies to help generate concise, professional support-response drafts. Every generated response is intended for **human review before being sent to a customer**.

## Project Status

Work in progress — core backend, RAG retrieval, and frontend integration are implemented. Gemini-based generation and evaluation remain to be completed and validated.

## Tech Stack

- Python
- Google Gemini API
- LangChain
- ChromaDB
- FastAPI
- Streamlit
- Pandas
- Pydantic

## Current Progress

- Kaggle IT support dataset prepared and normalized
- 500 English support conversations stored in the project dataset
- 10-example evaluation subset created
- Zero-shot, few-shot, and reasoned prompting implemented
- RAG prompting implemented
- Gemini embedding integration implemented
- ChromaDB vector store populated with 500 support conversations
- Similar-ticket retrieval implemented
- FastAPI backend implemented
- Mock mode implemented for development and testing without API usage
- API quota error handling implemented
- Responsible AI configuration added
- Streamlit frontend implemented
- Frontend-to-FastAPI integration tested successfully
- Retrieved RAG sources displayed in the frontend
- Human-review requirement incorporated into the application

## System Overview

```text
Customer Issue
      │
      ▼
  Streamlit UI
      │
      ▼
  FastAPI Backend
      │
      ├── Prompt Selection
      │
      ├── RAG Retrieval ──► ChromaDB
      │                         │
      │                         ▼
      │                  Similar Support Tickets
      │
      ▼
   Gemini API
      │
      ▼
 Draft Support Response
      │
      ▼
   Human Review
````

## Project Goal

The final system will allow a human support agent to:

1. Enter a customer's support issue.
2. Select a prompting strategy.
3. Retrieve relevant historical support conversations.
4. Generate a concise support-response draft.
5. Review the retrieved sources used as context.
6. Review the generated response before sending it to the customer.

## Responsible AI

The project includes responsible-AI guidelines covering:

* Accuracy and grounding
* Respectful language
* Risky or sensitive cases
* Privacy
* Mandatory human review

The assistant is designed as a **response-drafting tool**, not an autonomous customer-support agent.

## Remaining Work

* Complete Gemini baseline generation experiments
* Complete RAG generation experiments
* Run automated evaluation
* Generate evaluation results
* Complete the final project report
* Perform final end-to-end testing
* Final repository cleanup and documentation

