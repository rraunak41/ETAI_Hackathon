import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Industrial Knowledge Brain",
    page_icon="🏭",
)

load_dotenv()

st.title("🏭 Plant Operations & Asset Intelligence Brain")
st.caption(
    "Graph-Augmented RAG System for Industrial Maintenance, P&ID QA & Compliance"
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------
# PURE-PYTHON TEXT SPLITTER
# IMPORTANT:
# Do NOT import langchain_text_splitters here.
# That package/dependency chain can pull PyTorch into the
# environment, which is exactly what your Windows error shows.
# ---------------------------------------------------------
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Simple character-based splitter with no ML/PyTorch dependency."""
    if not text:
        return []

    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks


# ---------------------------------------------------------
# LAZY LOAD AI COMPONENTS
# FastEmbed uses ONNX rather than PyTorch for embeddings.
# ---------------------------------------------------------
@st.cache_resource
def load_rag_pipeline():
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_groq import ChatGroq

        embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        groq_key = os.getenv("GROQ_API_KEY")

        if not groq_key:
            return embeddings, None, None, (
                "GROQ_API_KEY is not set. Embeddings are ready, "
                "but the answer-generation model is unavailable."
            )

        llm = ChatGroq(
            temperature=0,
            model_name="openai/gpt-oss-20b",
            groq_api_key=groq_key,
        )

        vectorstore = None

        if Path(CHROMA_DIR).exists():
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )

        return embeddings, llm, vectorstore, None

    except Exception as e:
        return None, None, None, str(e)


# ---------------------------------------------------------
# INITIALIZE
# ---------------------------------------------------------
with st.spinner("Loading AI Embeddings & Vector Store..."):
    embeddings, llm, vectorstore, init_error = load_rag_pipeline()

if init_error:
    st.error(f"AI initialization error: {init_error}")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = vectorstore

if "llm" not in st.session_state:
    st.session_state.llm = llm

# ---------------------------------------------------------
# SIDEBAR: DATA INGESTION
# ---------------------------------------------------------
st.sidebar.header("📥 Custom Industry Data Ingestion")

uploaded_files = st.sidebar.file_uploader(
    "Upload Plant Manuals, Incident Logs, or PIDQA Datasets (.txt, .csv)",
    type=["txt", "csv"],
    accept_multiple_files=True,
)

if st.sidebar.button("Index Uploaded Files", type="primary"):
    if not uploaded_files:
        st.sidebar.warning("Please upload a .txt or .csv file first.")
    elif embeddings is None:
        st.sidebar.error("Embedding model is not available.")
    else:
        try:
            from langchain_core.documents import Document
            from langchain_community.vectorstores import Chroma

            documents = []

            for file in uploaded_files:
                filename = file.name.lower()

                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(file)
                    except Exception as csv_error:
                        st.sidebar.error(
                            f"Could not read {file.name}: {csv_error}"
                        )
                        continue

                    for idx, row in df.iterrows():
                        values = []

                        for col, val in row.items():
                            if pd.notna(val):
                                values.append(f"{col}: {val}")

                        row_text = " | ".join(values)

                        if row_text.strip():
                            documents.append(
                                Document(
                                    page_content=row_text,
                                    metadata={
                                        "source": file.name,
                                        "row": int(idx),
                                    },
                                )
                            )

                else:
                    try:
                        content = file.read().decode("utf-8", errors="replace")
                    except Exception as txt_error:
                        st.sidebar.error(
                            f"Could not read {file.name}: {txt_error}"
                        )
                        continue

                    if content.strip():
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={"source": file.name},
                            )
                        )

            if not documents:
                st.sidebar.warning("No readable content was found.")
            else:
                chunks = []

                for document in documents:
                    pieces = split_text(
                        document.page_content,
                        chunk_size=500,
                        chunk_overlap=50,
                    )

                    for piece_index, piece in enumerate(pieces):
                        metadata = dict(document.metadata)
                        metadata["chunk"] = piece_index

                        chunks.append(
                            Document(
                                page_content=piece,
                                metadata=metadata,
                            )
                        )

                with st.spinner("Creating embeddings and indexing data..."):
                    new_vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=CHROMA_DIR,
                    )

                st.session_state.vectorstore = new_vectorstore

                st.sidebar.success(
                    f"✅ Indexed {len(chunks)} chunks from "
                    f"{len(uploaded_files)} file(s)!"
                )

        except Exception as ingest_error:
            st.sidebar.error(
                f"Indexing failed: {ingest_error}"
            )

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
tab1, tab2 = st.tabs(
    ["💬 Operational Copilot", "🕸️ Dynamic Knowledge Graph"]
)

# ---------------------------------------------------------
# TAB 1: RAG QUESTION ANSWERING
# ---------------------------------------------------------
with tab1:
    st.subheader("Query Asset Maintenance & Compliance Corpus")

    user_query = st.text_input(
        "Ask a technical question regarding the uploaded dataset:",
        "How many valves or instruments are connected in the dataset?",
    )

    if st.button("Run Intelligence Query", type="primary"):
        current_vectorstore = st.session_state.get("vectorstore")
        current_llm = st.session_state.get("llm")

        if current_vectorstore is None:
            st.warning(
                "No Knowledge Base found. Upload a .txt or .csv file "
                "using the sidebar and click 'Index Uploaded Files'."
            )

        elif current_llm is None:
            st.error(
                "The vector database is available, but the answer-generation "
                "LLM is unavailable. Set GROQ_API_KEY in your .env file."
            )

        elif not user_query.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Analyzing maintenance logs and regulations..."
                ):
                    retriever = current_vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    docs = retriever.invoke(user_query)

                    if not docs:
                        st.warning(
                            "No relevant documents were found for this question."
                        )
                    else:
                        context_parts = []

                        for doc in docs:
                            source = doc.metadata.get("source", "Unknown")
                            context_parts.append(
                                f"Source ({source}):\n{doc.page_content}"
                            )

                        context = "\n\n".join(context_parts)

                        prompt = f"""
You are an expert industrial operations, P&ID analysis,
maintenance, and safety AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent facts.
2. If the context does not contain enough information, say so.
3. Keep the answer technically precise.
4. When possible, mention the source file.
5. For numerical questions, calculate only from the provided context.

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:
"""

                        response = current_llm.invoke(prompt)

                        st.markdown("### 🤖 AI Answer")
                        st.write(response.content)

                        st.markdown("---")
                        st.markdown("### 🔍 Source Citations")

                        for doc in docs:
                            source = doc.metadata.get(
                                "source", "Unknown"
                            )

                            preview = doc.page_content[:300]

                            st.info(
                                f"📄 **File:** `{source}`\n\n"
                                f"_{preview}..._"
                            )

            except Exception as query_error:
                st.error(
                    f"Query failed: {query_error}"
                )

# ---------------------------------------------------------
# TAB 2: KNOWLEDGE GRAPH
# ---------------------------------------------------------
with tab2:
    st.subheader("Interactive Knowledge Graph")

    try:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components

        G = nx.DiGraph()

        G.add_node(
            "Pump_P101",
            label="Pump P-101\n(Asset)",
            color="#3B82F6",
            shape="ellipse",
        )

        G.add_node(
            "Bearing_Wear",
            label="Bearing Fatigue\n(Root Cause)",
            color="#EF4444",
            shape="ellipse",
        )

        G.add_node(
            "BRG-6210-2RS",
            label="BRG-6210-2RS\n(Part #)",
            color="#F59E0B",
            shape="ellipse",
        )

        G.add_node(
            "OISD_118",
            label="OISD Standard 118\n(Regulation)",
            color="#10B981",
            shape="ellipse",
        )

        G.add_edge(
            "Pump_P101",
            "Bearing_Wear",
            title="EXHIBITED",
        )

        G.add_edge(
            "Bearing_Wear",
            "BRG-6210-2RS",
            title="REQUIRES_PART",
        )

        G.add_edge(
            "Pump_P101",
            "OISD_118",
            title="GOVERNED_BY",
        )

        net = Network(
            height="450px",
            width="100%",
            notebook=False,
            bgcolor="#111827",
            font_color="white",
        )

        net.from_nx(G)

        graph_path = "graph.html"
        net.save_graph(graph_path)

        with open(graph_path, "r", encoding="utf-8") as graph_file:
            html_content = graph_file.read()

        components.html(
            html_content,
            height=470,
            scrolling=False,
        )

    except Exception as graph_error:
        st.warning(
            f"Knowledge Graph rendering error: {graph_error}"
        )

# ---------------------------------------------------------
# FOOTER / DIAGNOSTICS
# ---------------------------------------------------------
with st.sidebar.expander("🔧 System Diagnostics"):
    st.write(
        f"**Embedding model:** `{EMBEDDING_MODEL}`"
    )
    st.write(
        f"**Vector DB:** `{CHROMA_DIR}`"
    )
    st.write(
        f"**Embeddings loaded:** "
        f"{'✅' if embeddings is not None else '❌'}"
    )
    st.write(
        f"**LLM loaded:** "
        f"{'✅' if llm is not None else '❌'}"
    )
