"""
===========================================================
STREAMLIT RAG PIPELINE (LOCAL + MULTI-FILE + RAGAS)
- Supports: PDF, TXT, DOCX
- Uses: ChromaDB + Ollama (NO API)
- Includes: RAGAS evaluation (local)
===========================================================
"""

import os
import streamlit as st

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
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Local RAG App", layout="wide")
st.title("🤖 Local RAG Pipeline (Ollama + Chroma + RAGAS)")

# =========================
# LOAD DOCUMENTS
# =========================

def load_documents(folder="data"):
    docs = []

    if not os.path.exists(folder):
        return docs

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
            st.warning(f"Error loading {file}: {e}")

    return docs

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Settings")
run_eval = st.sidebar.checkbox("Run RAGAS Evaluation")

# =========================
# LOAD + PROCESS
# =========================

@st.cache_resource
def setup_rag():
    documents = load_documents()

    if not documents:
        return None, None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )

    retriever = vector_db.as_retriever()

    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    Answer only using the context below.

    Context:
    {context}

    Question:
    {question}
    """)

    return retriever, llm, prompt

retriever, llm, prompt = setup_rag()

if retriever is None:
    st.error("❌ No documents found in 'data' folder")
    st.stop()

# =========================
# QUERY INPUT
# =========================

query = st.text_input("💬 Ask your question")

if query:
    with st.spinner("Retrieving context..."):
        retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt.format(context=context, question=query))

    st.subheader("🧠 Answer")
    st.write(response)

    # =========================
    # SHOW CONTEXT
    # =========================

    with st.expander("📚 Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:500])

    # =========================
    # RAGAS EVALUATION
    # =========================

    if run_eval:
        st.subheader("📊 RAGAS Evaluation")

        data = {
            "question": [query],
            "answer": [response],
            "retrieved_contexts": [[doc.page_content for doc in retrieved_docs]],
            "contexts": [[doc.page_content for doc in retrieved_docs]],
            "ground_truth": ["Provide correct expected answer here"]
        }

        dataset = Dataset.from_dict(data)

        eval_llm = LangchainLLMWrapper(Ollama(model="llama3"))

        eval_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="nomic-embed-text")
        )

        with st.spinner("Running evaluation..."):
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

        st.write(result)

st.success("✅ Ready")
