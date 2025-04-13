# 🧠 F.R.I.D.A.Y: Your Personal, Private, File-Aware AI Assistant

**FRIDAY** is a local-first, customizable AI assistant inspired by Iron Man's AI companion.  
It runs entirely on your machine, respects your privacy, supports multi-turn conversations, reads your personal files (ebooks, notes, logs), and learns from them using vector search + Retrieval Augmented Generation (RAG).

Built with Python, Streamlit, ChromaDB, HuggingFace embeddings, and Ollama-powered LLMs.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 🧵 Threaded chat sessions | Multi-turn conversations, saved per topic/thread |
| 📁 File upload + embedding | Upload `.pdf` and `.txt` files; auto-embedded into memory |
| 🧠 RAG-powered QA | Ask questions about your notes, ebooks, and documents |
| 🗃️ ChromaDB vector store | Fast, persistent vector search of embedded chunks |
| 🔁 Smart re-embedding | Automatically skips files already embedded (hash caching) |
| 📖 Chapter summarization | Summarize books and long texts by chapter |
| 📂 File-specific querying | Limit your query to one file or query all at once |
| 🧩 Cross-RAG linking | Get related content across multiple files intelligently |
| 🧑 Persona customization | Rename your assistant + define its behavior |
| 💬 Persistent chat memory | All messages and threads saved locally |
| 🔒 100% offline & private | All data stays on your machine; no external APIs |
| 📦 Modular & extensible | Add Whisper input, journaling agents, or goal trackers later |

---

## 📁 Project Structure
friday-ai/
├── app.py                      # Main Streamlit app
├── embed_utils.py             # Smart embedding logic w/ file hash detection
├── summarizer_utils.py        # Chapter splitting and summarizing
├── .gitignore                 # Ignore local files & DBs
├── requirements.txt           # All Python dependencies
│
├── db/                        # Chroma vector database (auto-generated)
├── my_context/
│   └── uploads/               # Your uploaded files go here
│
├── chat_threads.json          # Threaded chat memory
├── file_hash_cache.json       # Embedded file hash cache



---

## ⚙️ Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed with models like `phi3`, `mistral`, or `llama3`

---

## 🔧 Installation

```bash
git clone https://github.com/Ahmad-Waliullah-Khan/rag-based-personal-ai
cd rag-based-personal-ai
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```
---



## Privacy by Design
- ✅ 100% local-first

- ❌ No OpenAI keys

- ❌ No third-party cloud calls

- 🔒 All files, memory, and embeddings live inside your machine


## Contributions
Want to contribute? Suggest improvements, ideas, or fork and make it your own digital OS.