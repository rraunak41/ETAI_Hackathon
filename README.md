# 🏭 Asset & Operations Intelligence Brain
hii

> **Enterprise-Grade Graph-Augmented RAG Platform for Industrial Plant Safety, P&ID Question Answering & Maintenance Intelligence**

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-4c1?style=for-the-badge&logo=render)](https://plant-ai-brain.onrender.com/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/rraunak41/ETAI_Hackathon)

---

## 🌐 Live Deployment

* **Deployed Web Application:** [https://plant-ai-brain.onrender.com](https://plant-ai-brain.onrender.com)
* **Target Audience:** Plant Operators, Reliability Engineers, Safety Auditors, and Maintenance Technicians.

---

## 📌 Problem & Solution Overview

Heavy industrial facilities (refineries, power plants, chemical manufacturing) face critical information silos across 7–12 disconnected document stores—such as Piping and Instrumentation Diagrams (P&IDs), daily shift handovers, OEM maintenance manuals, and statutory safety codes (OISD, OSHA, Factory Acts). Field personnel spend up to 35% of their working hours searching for fragmented operational context.

The **Asset & Operations Intelligence Brain** resolves this by combining dense vector retrieval with an interactive **Entity Knowledge Graph**. It enables instant cross-referencing between equipment IDs, physical failure symptoms, recommended repair actions, and relevant regulatory clauses—backed by verbatim citations.

---

## ✨ Core Capabilities

* **💬 Low-Latency Operational Copilot:** Powered by Meta's `Llama-3.1-8b-instant` via Groq LPU inference for high-speed technical question answering.
* **🕸️ Dynamic Knowledge Graph:** Visualizes cross-entity relationships (e.g., `Pump P-101` $\rightarrow$ `Bearing Fatigue` $\rightarrow$ `BRG-6210-2RS` $\rightarrow$ `OISD Standard 118`) using NetworkX and interactive PyVis canvas rendering.
* **📥 Multi-Modal Ingestion Engine:** Dynamic file indexing for unstructured plant incident logs (`.txt`) and structured tabular datasets (`.csv` P&ID QA sets).
* **🔍 Audit-Ready Source Grounding:** Returns explicit source filenames and chunk citations with every AI output to prevent hallucinations in safety-critical operations.

---

## 🏗️ System Architecture

```text
               [ Uploaded Logs (.txt) / Datasets (.csv) ]
                                   │
                                   ▼
             [ Recursive Character Chunking & Parsing ]
                                   │
                                   ▼
          [ Dense Embeddings Engine (all-MiniLM-L6-v2) ]
                                   │
                                   ▼
          [ Local ChromaDB Vector Store ] ◄───► [ NetworkX Entity Graph ]
                                   │
                                   ▼
                     [ Top-K Hybrid Context Assembly ]
                                   │
                                   ▼
                   [ Meta Llama-3.1 (Groq API) ]
                                   │
                                   ▼
             [ Streamlit Copilot UI + Visual Graph Display ]
