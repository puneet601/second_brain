# 🧠 Second Brain – Multi-Agent System (MAS)

## 📘 Overview

The **Second Brain MAS** is a **multi-agent system** designed to simulate autonomous reasoning, task delegation, and research capabilities.
It integrates **retrieval-augmented generation (RAG)**, **controller-driven orchestration**, and **event-driven design (EDD)** to create an intelligent system.

---

## ⚙️ Architecture
```plaintext
second_brain/
├── core/                     # Core logic of all agents
│   ├── controller_agent.py   # Controls decision routing
│   ├── orchestrator.py       # Handles multi-agent communication
│   ├── researcher.py         # Fetches relevant notes
│   ├── synthesiser.py        # Summarizes and structures information
│   ├── preference_detector.py# Identifies user preferences and stores them
│   ├── RAGSystem.py          # Core retrieval-augmented generator
│   ├── conversation_memory.py# Maintains short- and long-term context
│   ├── guardrails.py         # Ensures safety and adherence to response rules (e.g., PII data)
│   ├── utils.py              # Helper utilities for reading/writing data
│   └── logger.py             # Custom logging system
│
├── evaluation/               # Evaluation and benchmarking framework
│   ├── baseline_bot.py       # Simple baseline model for comparison
│   └── baseline_evals.py     # Automated evaluation script for main functionalities
│
├── data/
│   └── memory/               # Persistent memory and logs
│
├── notes/                    # Knowledge base (used by RAG system)
│   ├── indian_recipe.txt
│   ├── movie.txt
│   └── plants.txt
│
├── main.py                   # Entry point for system execution
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Build configuration
└── uv.lock                   # Dependency lockfile
```
## 🧩 Agent Roles

### 1. ControllerAgent

* Acts as the system’s “brain.”
* Interprets user input and routes requests to appropriate agents.
* Maintains decision accuracy using a reasoning model.

### 2. Researcher

* Fetches contextually relevant notes using embedding similarity.
* Converts user queries into search embeddings.
* Uses `SentenceTransformer` for semantic retrieval.

### 3. Synthesiser

* Combines insights from multiple sources into concise, readable summaries.

### 4. PreferenceDetector

* Detects patterns in user language (e.g., likes, dislikes, interests).
* Updates persistent memory with personalized context.

### 5. RAGSystem

* Loads, chunks, and embeds notes from the knowledge base.
* Enables fast and relevant retrieval during user interactions.

---

## 🧮 Evaluation Results

| Model               | Average Relevance | Accuracy | Latency (s) |
| ------------------- | ----------------: | -------: | ----------: |
| **BaselineBot**     |              0.22 |        – |        0.00 |
| **RAGSystem**       |              0.83 |        – |        0.04 |
| **ControllerAgent** |          **1.00** |   ✅ 100% |        4.35 |

### 🧠 Insights

* The **ControllerAgent** perfectly classified all tested scenarios (`preference_query`, `research_task`, `quit_command`).
* The **RAGSystem** significantly outperformed the baseline, showing robust semantic retrieval capabilities.
* Latency is within acceptable range for multi-step reasoning tasks.

---

## 🚀 Running the System

### Prerequisites

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run the MAS

```bash
uv run python -m core.orchestrator
```

### Evaluate Performance

```bash
uv run python -m evaluation.baseline_evals
```
### Observability - Otle_tui with Logfire
![alt text](<Screenshot 2025-11-03 at 3.15.17 PM.png>)

---

## 📈 Future Work

* Introduce a **Planner Agent** for multi-step reasoning.
* Integrate **web retrieval** and **memory reinforcement learning**.
* Build a **frontend interface** for visualization and chat control.

---

## 👩‍💻 Author

**Puneet Jattana**
Consultant Developer | Thoughtworks
Focused on multi-agent reasoning systems, information retrieval, and applied AI research.
