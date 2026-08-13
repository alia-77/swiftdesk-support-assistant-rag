# SwiftDesk IT Support Assistant

## 1. Project Overview

SwiftDesk is a RAG-based IT support assistant designed to help support agents draft customer responses. The system retrieves similar historical support tickets from ChromaDB and provides their approved replies as context for Gemini. The generated response is treated as a draft and must be reviewed by a human before being sent.

## 2. Dataset Preparation

The project started with a Kaggle IT support dataset. The relevant English support conversations were cleaned and normalized into a dataset of **500 support examples** containing customer issues and reference replies.

A separate test subset of **10 customer issues** was created for evaluating the different prompting approaches.

## 3. Prompting Approaches

Three baseline prompting approaches were tested:

* **Zero-shot:** Gemini was given the customer issue and asked to write a support reply without examples.
* **Few-shot:** Gemini was given example customer issues and replies as guidance.
* **Reasoned:** Gemini was asked to analyze the issue before writing the response, without exposing its internal reasoning.

A fourth approach was then tested using **RAG**, where relevant historical support tickets were retrieved and provided as context.

## 4. RAG Implementation

The 500 support conversations were embedded using Gemini and stored in **ChromaDB**.

When a customer submits an issue, the system creates an embedding for the query and searches ChromaDB for the most similar support tickets. The retrieved customer issues and approved replies are then included in the prompt sent to Gemini.

The application uses **LangChain** for the retrieval layer, while FastAPI handles the backend API and Streamlit provides the user interface.

## 5. System Architecture

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
      ├── RAG Retrieval
      │       │
      │       ▼
      │    ChromaDB
      │       │
      │       ▼
      │  Similar Tickets
      │
      ▼
   Gemini API
      │
      ▼
 Draft Response
      │
      ▼
 Human Review
```

## 6. Responsible AI

The project includes guidelines for accuracy, respectful language, privacy, risky cases, and mandatory human review.

The assistant is designed to help a support agent draft a response rather than act as an autonomous support system. This is important because generated responses can contain unsupported information, especially when no relevant context is available.

## 7. Evaluation

The 10 test examples were evaluated using **ROUGE-L**, which measures overlap between the generated response and the reference reply.

The final average scores were:

| Approach  | Average ROUGE-L |
| --------- | --------------: |
| Zero-shot |          0.1637 |
| Few-shot  |          0.3113 |
| Reasoned  |          0.1647 |
| RAG       |      **0.9516** |

Few-shot prompting performed best among the three baseline approaches. RAG achieved a much higher score than the baselines.

## 8. Analysis

The results show that providing relevant historical support examples had a strong effect on the generated responses. The few-shot approach already improved the average score compared with zero-shot and reasoned prompting.

The RAG approach performed especially well because the ChromaDB collection contains support conversations that are very similar to several of the test cases. In some cases, the top retrieved document was the same customer issue or a very close variation of it, which allowed Gemini to produce a response very close to the reference reply.

Because of this, the high RAG score should not be interpreted as proof that the system would achieve the same performance on completely unseen customer issues. A larger test set containing more novel queries would provide a stronger evaluation.

## 9. Limitations

The evaluation used only 10 test examples, so the results should be treated as an initial comparison rather than a complete benchmark.

ROUGE-L also measures textual overlap and does not fully measure whether a response is helpful, accurate, or appropriate. Human evaluation would provide additional information about response quality.

The system also depends on the availability and rate limits of the Gemini API.

## 10. Conclusion

SwiftDesk demonstrates how a RAG-based support assistant can use historical support conversations to generate grounded response drafts.

The experiments showed that few-shot prompting performed better than zero-shot and reasoned prompting on the selected test set, while RAG produced the highest ROUGE-L score by using relevant historical support information.

The final system combines Gemini, ChromaDB, LangChain, FastAPI, and Streamlit into a simple support-drafting workflow with human review as the final step.
