# SwiftDesk IT Support Assistant

A RAG-based IT support assistant for drafting customer support responses using relevant historical support conversations as context.

The system retrieves similar support tickets from a ChromaDB vector store and uses their approved replies to help generate concise, professional response drafts. Every generated response is intended for **human review before being sent to a customer**.

## Project Overview

SwiftDesk was built to explore different prompting strategies and compare them with a retrieval-augmented generation (RAG) approach.

The system uses a dataset of 500 English IT support conversations. Similar historical tickets are retrieved from ChromaDB and provided to Gemini as context when RAG is enabled.

## Tech Stack

* Python
* Google Gemini API
* LangChain
* ChromaDB
* FastAPI
* Streamlit
* Pandas
* Pydantic

## System Architecture

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
```

## Dataset

The project uses a Kaggle IT support dataset that was cleaned and normalized into:

* **500 English support conversations**
* **10 evaluation examples**

Each example contains a customer issue and a reference support reply.

## Prompting Approaches

Four approaches were evaluated:

### Zero-shot

Gemini receives only the customer issue and is asked to produce a support response.

### Few-shot

Gemini receives example customer issues and reference replies as guidance before generating the response.

### Reasoned

Gemini is instructed to analyze the customer's issue before drafting the response without exposing its internal reasoning.

### RAG

The system retrieves similar historical support conversations from ChromaDB and provides the retrieved information to Gemini as context before generating the response.

## Evaluation

The four approaches were evaluated on the same 10 customer issues using **ROUGE-L**.

| Approach  | Average ROUGE-L |
| --------- | --------------: |
| Zero-shot |          0.1637 |
| Few-shot  |          0.3113 |
| Reasoned  |          0.1647 |
| **RAG**   |      **0.9516** |

Few-shot prompting performed best among the baseline approaches.

RAG produced the highest score because the retrieved support conversations were often highly similar to the test issues, and in several cases the dataset contained an exact or near-duplicate issue. Therefore, the high RAG score should not be interpreted as proof that the same performance would be achieved on completely unseen customer questions.

ROUGE-L measures textual overlap with the reference reply, so it is useful for comparison but does not fully measure response quality, correctness, or usefulness.

## Responsible AI

The project includes guidelines covering:

* Accuracy and grounding
* Respectful language
* Risky or sensitive cases
* Privacy
* Mandatory human review

SwiftDesk is designed as a **response-drafting assistant**, not an autonomous customer-support system.

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file based on `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key
MOCK_MODE=false
```

Never commit the real `.env` file.

### 3. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Start the Streamlit frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

## Project Structure

```text
swiftdesk-support-assistant-rag/
├── task_description.html
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── src/
│   ├── settings.py
│   ├── prompts.py
│   ├── rag_chain.py
│   ├── ingest_chroma.py
│   ├── prepare_kaggle_dataset.py
│   ├── gemini_basics.py
│   └── evaluation_script.py
├── data/
│   ├── raw/
│   ├── support_conversations.csv
│   └── test_subset.json
├── config/
│   └── RAI_Config.yaml
├── outputs/
│   ├── baseline_outputs.json
│   ├── evaluation_results.json
│   └── final_report.md
├── chroma_db/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Human Review

Generated responses should always be reviewed by a human support agent before being sent to a customer. The system is intended to assist support staff, not replace their judgment.
