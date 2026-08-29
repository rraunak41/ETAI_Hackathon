🏭 Asset & Operations Intelligence Brain
Enterprise-Grade Graph-Augmented RAG Platform for Industrial Plant Safety, P&ID Question Answering & Maintenance Intelligence
🌐 Live Deployment
Deployed Web Application: https://plant-ai-brain.onrender.com
Target Audience: Plant Operators, Reliability Engineers, Safety Auditors, and Maintenance Technicians.
📌 Problem & Solution OverviewHeavy industrial facilities (refineries, power plants, chemical manufacturing) face critical information silos across 7–12 disconnected document stores—such as Piping and Instrumentation Diagrams (P&IDs), daily shift handovers, OEM maintenance manuals, and statutory safety codes (OISD, OSHA, Factory Acts). Field personnel spend up to 35% of their working hours searching for fragmented operational context.
The Asset & Operations Intelligence Brain resolves this by combining dense vector retrieval with an interactive Entity Knowledge Graph. It enables instant cross-referencing between equipment IDs, physical failure symptoms, recommended repair actions, and relevant regulatory clauses—backed by verbatim citations.
✨ Core Capabilities
💬 Low-Latency Operational Copilot: Powered by Meta's Llama-3.1-8b-instant via Groq LPU inference for high-speed technical question answering.
🕸️ Dynamic Knowledge Graph: Visualizes cross-entity relationships (e.g., Pump P-101 $\rightarrow$ Bearing Fatigue $\rightarrow$ BRG-6210-2RS $\rightarrow$ OISD Standard 118) using NetworkX and interactive PyVis canvas rendering.
📥 Multi-Modal Ingestion Engine: Dynamic file indexing for unstructured plant incident logs (.txt) and structured tabular datasets (.csv P&ID QA sets).
🔍 Audit-Ready Source Grounding: Returns explicit source filenames and chunk citations with every AI output to prevent hallucinations in safety-critical operations.
🏗️ System ArchitecturePlaintext   
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
             
🛠️ Technology Stack
LayerTechnologies UsedLanguage & Web FrameworkPython 3.10+, StreamlitLLM InferenceMeta Llama-3.1-8b-instant (Groq Cloud API)Embeddings & Vector StoreHuggingFace all-MiniLM-L6-v2, ChromaDBGraph VisualizationNetworkX, PyVis, HTML ComponentsOrchestration & SplittingLangChain, LangChain Core, LangChain CommunityData ProcessingPandas, DotenvDeployment PlatformRender (PaaS), GitHub CI/CD🚀 Local Development Setup1. Clone the RepositoryBashgit clone https://github.com/rraunak41/ETAI_Hackathon.git
cd ETAI_Hackathon
2. Configure Virtual EnvironmentBash# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
4. Configure Environment VariablesCreate a .env file in the root directory:Code snippetGROQ_API_KEY=gsk_your_groq_api_key_here
5. Launch the Local ServerBashstreamlit run app.py
Access the application locally at http://localhost:8501.
🧪 Verification & Sample Queries
Once documents or P&ID datasets are uploaded and indexed via the sidebar:Asset Failure Diagnostics: "What caused the vibration spike on Pump P-101 and what replacement part is specified?"Compliance & Standards: "Which OISD standard applies to centrifugal pump maintenance?"P&ID Dataset Verification: "How many valves or instruments are spatially connected to line segment A-102?"
👥 Contributors
Raunak Kumar Jaiswal — GitHub Profile
