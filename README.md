# SwiftDesk IT Support Assistant

A RAG-based IT support assistant that retrieves similar historical support tickets and uses their approved replies as context for generating short support-response drafts.

## Current Status

Work in progress.

### Tech Stack

* Python
* Gemini API
* LangChain
* ChromaDB
* FastAPI
* Streamlit
* Pandas

### Current Progress

* Kaggle IT support dataset prepared
* 500 English support conversations normalized
* 10-example evaluation subset created
* Gemini prompt experiment structure created
* Chroma ingestion implemented
* RAG retrieval layer implemented
* FastAPI backend skeleton implemented
* Health endpoint tested successfully

### Project Goal

The final system will allow a human support agent to enter a customer issue, retrieve relevant previous support tickets, and generate a draft response for human review.
