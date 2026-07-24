import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

st.set_page_config(layout="wide", page_title="Industrial Knowledge Brain", page_icon="🏭")

st.title("🏭 Plant Operations & Asset Intelligence Brain")
st.caption("Graph-Augmented RAG System for Industrial Maintenance, P&ID QA & Compliance")

# Verify API Key early
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("🔑 GROQ_API_KEY not found! Please check your `.env` file.")
    st.stop()

# Lazy imports to avoid startup hangs
@st.cache_resource
def load_rag_pipeline():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain_community.vectorstores import Chroma

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_key)
        
        vectorstore = None
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            
        return embeddings, llm, vectorstore
    except Exception as e:
        st.error(f"Error initializing AI modules: {e}")
        return None, None, None

with st.spinner("Loading AI Embeddings & Vector Store..."):
    embeddings, llm, vectorstore = load_rag_pipeline()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = vectorstore

# ---------------------------------------------------------
# SIDEBAR: DATA INGESTION (Supports .txt and .csv)
# ---------------------------------------------------------
st.sidebar.header("📥 Custom Industry Data Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Plant Manuals, Incident Logs, or PIDQA Datasets (.txt, .csv)", 
    type=["txt", "csv"], 
    accept_multiple_files=True
)

if st.sidebar.button("Index Uploaded Files", type="primary"):
    if uploaded_files:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from langchain_community.vectorstores import Chroma

        documents = []
        for file in uploaded_files:
            if file.name.endswith(".csv"):
                # Convert CSV rows into textual descriptions
                df = pd.read_csv(file)
                for idx, row in df.iterrows():
                    row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    documents.append(Document(page_content=row_str, metadata={"source": file.name, "row": idx}))
            else:
                content = file.read().decode("utf-8")
                documents.append(Document(page_content=content, metadata={"source": file.name}))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        st.session_state.vectorstore = Chroma.from_documents(
            chunks, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        st.sidebar.success(f"✅ Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")
    else:
        st.sidebar.warning("Please upload a file first.")

# ---------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["💬 Operational Copilot", "🕸️ Dynamic Knowledge Graph"])

with tab1:
    st.subheader("Query Asset Maintenance & Compliance Corpus")
    user_query = st.text_input(
        "Ask a technical question regarding the uploaded dataset:", 
        "How many valves or instruments are connected in the dataset?"
    )
    
    if st.button("Run Intelligence Query", type="primary"):
        if st.session_state.vectorstore is None:
            st.warning("No Knowledge Base found! Upload a `.txt` or `.csv` file using the sidebar to index data.")
        else:
            with st.spinner("Analyzing maintenance logs and regulations..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(user_query)
                context = "\n\n".join([f"Source ({doc.metadata.get('source', 'Doc')}):\n" + doc.page_content for doc in docs])
                
                prompt = f"""You are an expert industrial operations, P&ID diagram analysis, and safety AI assistant. 
Answer the technical question using ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:"""
                
                response = llm.invoke(prompt)
                st.markdown("### **AI Answer:**")
                st.write(response.content)
                
                st.markdown("---")
                st.markdown("### 🔍 **Source Citations:**")
                for doc in docs:
                    st.info(f"📄 **File:** `{doc.metadata.get('source', 'Doc')}`\n\n_{doc.page_content[:250]}..._")

with tab2:
    st.subheader("Interactive Knowledge Graph")
    try:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components

        G = nx.DiGraph()
        G.add_node("Pump_P101", label="Pump P-101\n(Asset)", color="#3B82F6", shape="ellipse")
        G.add_node("Bearing_Wear", label="Bearing Fatigue\n(Root Cause)", color="#EF4444", shape="ellipse")
        G.add_node("BRG-6210-2RS", label="BRG-6210-2RS\n(Part #)", color="#F59E0B", shape="ellipse")
        G.add_node("OISD_118", label="OISD Standard 118\n(Regulation)", color="#10B981", shape="ellipse")

        G.add_edge("Pump_P101", "Bearing_Wear", title="EXHIBITED")
        G.add_edge("Bearing_Wear", "BRG-6210-2RS", title="REQUIRES_PART")
        G.add_edge("Pump_P101", "OISD_118", title="GOVERNED_BY")

        net = Network(height="450px", width="100%", notebook=False, bgcolor="#111827", font_color="white")
        net.from_nx(G)
        net.save_graph("graph.html")
        
        with open("graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=470)
    except Exception as graph_err:
        st.warning(f"Knowledge Graph rendering error: {graph_err}")