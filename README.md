# 📦 RAG Pipeline (Multi-Document + Evaluation)

A **Retrieval-Augmented Generation (RAG) system** built using Open source models.  
This project demonstrates how to build an end-to-end pipeline that:

- Ingests multiple document formats (PDF, TXT, DOCX)
- Performs semantic retrieval using vector embeddings
- Generates context-aware answers using an LLM
- Evaluates response quality using RAGAS metrics

---

# 🚀 Key Features

- ✅ Multi-format document ingestion (PDF, TXT, DOCX)
- ✅ Chunking strategy for optimal retrieval
- ✅ Dense vector search using embeddings
- ✅ Local LLM inference via Ollama
- ✅ Persistent vector storage with ChromaDB
- ✅ Built-in evaluation using RAGAS
- ✅ Clean, modular pipeline (easy to extend)

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances LLM responses by grounding them in external data.

### Standard LLM ❌
- Generates answers from pre-trained knowledge only

### RAG System ✅
1. Retrieves relevant information from documents
2. Uses that context to generate accurate responses

---

## 🔄 Pipeline Overview

```
User Query
    ↓
Embedding Model
    ↓
Vector Similarity Search (Retriever)
    ↓
Relevant Context Chunks
    ↓
LLM (Answer Generation)
    ↓
Final Response
    ↓
RAGAS Evaluation
```

---

# 🔍 Retrieval Methodology

This project uses:

### ✅ Dense Vector Similarity Search

- Embedding Model: `nomic-embed-text`
- Vector Store: `ChromaDB`
- Retrieval: Top-K similarity search

---

## 🔄 Symmetric vs Asymmetric Search

| Type | Description | Used |
|------|------------|------|
| Symmetric | Query and documents are similar in size/style | ❌ |
| Asymmetric | Short query vs long documents | ✅ |

👉 This implementation uses **Asymmetric Semantic Search**, which is ideal for:
- Question-answering systems
- Document retrieval use cases

---

# 🏗️ Project Structure

```
rag-project/
│
├── main.py                  # Main RAG pipeline
│
├── data/                   # Input documents
│   ├── sample.pdf
│   ├── notes.txt
│   ├── report.docx
│
├── chroma_db/              # Persistent vector store
│
├── requirements.txt        # Dependencies
│
└── README.md               # Project documentation
```

---

# ⚙️ System Components

## 1️⃣ Document Loader

Supports:
- PDF (`PyPDFLoader`)
- TXT (`TextLoader`)
- DOCX (`Docx2txtLoader`)

---

## 2️⃣ Text Chunking

```
chunk_size = 500
chunk_overlap = 50
```

Ensures:
- Better semantic coherence
- Improved retrieval accuracy

---

## 3️⃣ Embeddings

- Model: `nomic-embed-text`
- Type: Dense semantic embeddings
- Converts text → vector representations

---

## 4️⃣ Vector Store

- Database: `ChromaDB`
- Stores:
  - Text chunks
  - Corresponding embeddings
- Persistence enabled (`chroma_db/`)

---

## 5️⃣ Retriever

```
retriever = vector_db.as_retriever()
```

- Performs similarity search
- Returns top-K relevant chunks

---

## 6️⃣ LLM (Local Inference)

- Model: `llama3` (via Ollama)
- Role:
  - Context-aware answer generation
  - Evaluation (via RAGAS wrapper)

---

## 7️⃣ Prompt Design

```
Answer only using the context below.
```

- Reduces hallucination
- Ensures grounded responses

---

# 📊 Evaluation with RAGAS

This project integrates **RAGAS** for systematic evaluation.

## Metrics Used

| Metric | Description |
|--------|------------|
| context_precision | Relevance of retrieved chunks |
| context_recall | Coverage of relevant information |
| faithfulness | Consistency with context |
| answer_relevancy | Quality of answer |

---

## Dataset Format

```
{
  question,
  answer,
  retrieved_contexts,
  contexts,
  ground_truth
}
```

> ⚠️ Replace `ground_truth` with expected answers for meaningful evaluation

---

# 🚀 Getting Started

## 1️⃣ Install Ollama

Download and install ollama

---

## 2️⃣ Pull Required Models

```
ollama pull llama3
ollama pull nomic-embed-text
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 4️⃣ Add Documents

Place your files inside:

```
/data
```

---

## 5️⃣ Run the Application

---


# 🧠 Key Concepts

| Concept | Description |
|--------|------------|
| Embeddings | Vector representation of text |
| Chunking | Splitting documents for processing |
| Retriever | Finds relevant content |
| Vector DB | Stores embeddings |
| LLM | Generates responses |
| RAGAS | Evaluates system quality |

---

# ⚠️ Limitations

- No keyword-based retrieval (BM25)
- No hybrid search (dense + sparse)
- No re-ranking layer
- Performance depends on embedding quality

---

# 🚀 Future Enhancements

- Hybrid Search (BM25 + Vector)
- Cross-encoder re-ranking
- Metadata filtering
- Query expansion techniques
- REST API (FastAPI)
---

# 🏁 Conclusion

This project demonstrates a **complete local RAG architecture**, combining:

- Dense semantic retrieval  
- Context-aware LLM generation  
- Quantitative evaluation  

👉 A strong foundation for building production-grade AI applications.

---

# 📬 Use Cases

- Enterprise document search  
- Knowledge base assistants  
- Internal Q&A systems  
- Research assistants  

---

## This project was developed as part of my learning journey in Python. As a learning assistant to understand concepts and structure the code took help of GPT Models. The final implementation, testing, and project setup were completed by "infoanupampal@gmail.com"


<div align="center">

Built with ❤️ by **anupamLab**

</div>
