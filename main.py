"""
===========================================================
FINAL RAG PIPELINE (LOCAL + MULTI-FILE + RAGAS)
- Supports: PDF, TXT, DOCX
- Uses: ChromaDB + Ollama (NO API)
- Includes: RAGAS evaluation (local)
===========================================================
"""

# =========================
# 1. IMPORTS
# =========================

import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# =========================
# 2. LOAD DOCUMENTS
# =========================

def load_documents(folder="data"):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            if file.endswith(".txt"):
                docs.extend(TextLoader(path).load())

            elif file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())

            elif file.endswith(".docx"):
                docs.extend(Docx2txtLoader(path).load())

        except Exception as e:
            print(f"Error loading {file}: {e}")

    return docs


print("Loading documents...")
documents = load_documents()

if not documents:
    raise ValueError("No documents found in 'data' folder!")


# =========================
# 3. CHUNKING
# =========================

print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# =========================
# 4. VECTOR STORE
# =========================

print("Creating vector database...")
embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

retriever = vector_db.as_retriever()


# =========================
# 5. LLM
# =========================

print("Loading LLM...")
llm = Ollama(model="llama3")   # better than mistral for eval

prompt = ChatPromptTemplate.from_template("""
Answer only using the context below.

Context:
{context}

Question:
{question}
""")


# =========================
# 6. ASK QUESTION
# =========================

query = input("\nAsk your question: ")

print("Retrieving context...")
retrieved_docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in retrieved_docs)

print("Generating answer...")
response = llm.invoke(prompt.format(context=context, question=query))

print("\n================ ANSWER ================\n")
print(response)


# =========================
# 7. RAGAS EVALUATION (FINAL FIX)
# =========================

print("\nRunning evaluation...")

data = {
    "question": [query],
    "answer": [response],

    # REQUIRED FOR CONTEXT METRICS
    "retrieved_contexts": [[doc.page_content for doc in retrieved_docs]],

    # REQUIRED FOR FAITHFULNESS
    "contexts": [[doc.page_content for doc in retrieved_docs]],

    # REQUIRED FOR RECALL
    "ground_truth": ["Provide correct expected answer here"]
}

dataset = Dataset.from_dict(data)

# Local evaluation models
eval_llm = LangchainLLMWrapper(Ollama(model="llama3"))

eval_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ],
    llm=eval_llm,
    embeddings=eval_embeddings
)

print("\n================ EVALUATION ================\n")
print(result)

print("\nDone ✅")