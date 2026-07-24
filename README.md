# 🏭 Asset & Operations Intelligence Brain

> **ET AI Hackathon 2026 Submission** | *Problem Statement 8: AI for Industrial Knowledge Intelligence*

An enterprise-grade **Graph-Augmented Retrieval-Augmented Generation (GraphRAG)** platform designed to eliminate operational information fragmentation, perform automated P&ID diagram question-answering, and digitize critical industrial knowledge for plant safety and compliance.

---

## 📌 Executive Summary

Heavy industrial facilities (refineries, power plants, chemical units) often operate across 7 to 12 disconnected document stores—including P&IDs, daily maintenance logs, OEM manuals, and OISD/Factory Act regulatory standards. Field engineers spend over 35% of their time searching for or verifying operational context, leading to prolonged unplanned downtime and compliance risks.

The **Asset & Operations Intelligence Brain** bridges this gap by unifying unstructured document search with an interactive **Entity Knowledge Graph**, delivering instant, zero-hallucination operational diagnostics backed by explicit document citations.

---

## ✨ Key Features

* **💬 Operational Knowledge Copilot:** Resolves complex operational, P&ID, and maintenance queries with low latency using Meta's `Llama-3.1-8b` via the Groq API.
* **🕸️ Dynamic Knowledge Graph:** Visualizes non-linear relationships across equipment IDs (`Pump_P101`), root-cause failure modes (`Bearing Fatigue`), replacement part numbers (`BRG-6210-2RS`), and safety regulations (`OISD Standard 118`).
* **📥 Multi-Format Ingestion Engine:** Supports instant sidebar upload and dynamic vector indexing of both `.txt` plant logs and structured tabular dataset files (`.csv` P&ID corpora).
* **🔍 Audit-Ready Citation Cards:** Displays exact source filenames and text passage excerpts alongside generated answers for complete auditability.

---

## 🛠️ Tech Stack

* **LLM Engine:** Meta Llama-3.1-8b-instant (Groq API)
* **Embedding Model:** `all-MiniLM-L6-v2` (`HuggingFaceEmbeddings` / `sentence-transformers`)
* **Vector Store:** ChromaDB
* **Knowledge Graph Engine:** NetworkX & PyVis
* **Orchestration Framework:** LangChain
* **User Interface:** Streamlit

---

## 🏗️ System Architecture

```text
[ Unstructured Documents (.txt) / Datasets (.csv) ]
                         │
                         ▼
           [ Recursive Chunking Engine ]
                         │
                         ▼
     [ Dense Embeddings (all-MiniLM-L6-v2) ]
                         │
                         ▼
           [ Local Chroma Vector Store ] ◄───► [ NetworkX Entity Graph ]
                         │
                         ▼
           [ Hybrid Retriever Context ]
                         │
                         ▼
            [ Llama-3.1-8b (Groq API) ]
                         │
                         ▼
         [ Streamlit UI & Source Citations ]
