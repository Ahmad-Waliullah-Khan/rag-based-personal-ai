import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import os
import json
import time
import os


# === CONFIG ===
CHAT_LOG_FILE = "chat_history.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "db"
MAX_HISTORY_CONTEXT = 2

# === DEFAULT ASSISTANT CONFIG ===
DEFAULT_NAME = "Friday"
DEFAULT_PERSONA = (
    "You are a calm, insightful and highly intelligent personal assistant. "
    "You remember previous conversations, help me reflect, plan, and guide decisions with empathy and data."
)

# === PAGE CONFIG ===
st.set_page_config(page_title="🧠 F.R.I.D.A.Y ", layout="wide")
st.title("🧠 F.R.I.D.A.Y - Ahmad's personal ambient OS")

from datetime import datetime

greeting = "Good morning" if datetime.now().hour < 12 else (
    "Good afternoon" if datetime.now().hour < 18 else "Good evening"
)

# === SESSION STATE INIT ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "assistant_name" not in st.session_state:
    st.session_state.assistant_name = DEFAULT_NAME
if "assistant_persona" not in st.session_state:
    st.session_state.assistant_persona = DEFAULT_PERSONA

# === LOAD + SAVE CHAT MEMORY ===
def load_chat_history():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)

# Load persistent memory on startup
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history()

st.markdown(f"### 👋 {greeting}, Mr. Khan")
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    st.markdown(f"Last time, you asked about **{last['topic']}**.")
else:
    st.markdown(f"Your assistant **{st.session_state.assistant_name}** is ready.")


# === TOPIC DETECTION ===
def infer_topic(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["finance", "money", "investment", "sip", "saving"]):
        return "Finance"
    elif any(x in q for x in ["goal", "project", "plan", "milestone", "dream"]):
        return "Goals"
    elif any(x in q for x in ["health", "mental", "doctor", "therapy", "stress"]):
        return "Health"
    elif any(x in q for x in ["journal", "diary", "life", "feeling", "emotion"]):
        return "Diary"
    return "General"

# === SIDEBAR CONFIG ===
with st.sidebar:
    st.markdown("### 🧑 Assistant Settings")
    st.session_state.assistant_name = st.text_input("🤖 Assistant Name", value=st.session_state.assistant_name)
    st.session_state.assistant_persona = st.text_area(
        "🎭 Assistant Persona",
        value=st.session_state.assistant_persona,
        height=120
    )

    # Topic filter
    st.markdown("### 💬 Persistent Chat History")
    topics = sorted(set(item["topic"] for item in st.session_state.chat_history))
    selected_topic = st.selectbox("🗂️ Filter by Topic", ["All"] + topics)

    # Chat history display
    for entry in reversed(st.session_state.chat_history):
        if selected_topic != "All" and entry["topic"] != selected_topic:
            continue
        st.markdown(f"**🕒 {entry['time']}**")
        st.markdown(f"**🗂️ {entry['topic']}**")
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**{st.session_state.assistant_name}:** {entry['response']}")
        st.markdown("---")

    if st.button("🧹 Clear All Memory"):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("Memory cleared.")

# === MODEL SELECTION ===
model_choice = st.selectbox("🔧 Choose your model", ["phi3:mini", "llama3", "mistral"])


def stream_text(text):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(f"```\n{typed}▌\n```")
        time.sleep(0.015)
    placeholder.markdown(f"```\n{typed}\n```")

# === CHAIN LOADER ===
@st.cache_resource
def load_qa_chain(selected_model):
    llm = Ollama(model=selected_model)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = load_qa_chain(model_choice)

# === INPUT ===
query = st.text_input("💬 Ask your assistant...")

def speak(text):
    os.system(f'say -v Kate "{text}"')

if query:
    # === Construct prompt with persona + history ===
    recent_turns = st.session_state.chat_history[-MAX_HISTORY_CONTEXT:]
    context = "\n".join(
        f"You: {turn['query']}\n{st.session_state.assistant_name}: {turn['response']}"
        for turn in recent_turns
    )

    persona_header = (
        f"{st.session_state.assistant_name}, here is your role:\n"
        f"{st.session_state.assistant_persona}\n\n"
    )

    full_query = persona_header
    if context:
        full_query += f"{context}\n\nNow respond to: {query}"
    else:
        full_query += f"User says: {query}"

    with st.spinner("Thinking..."):
        response = qa_chain.run(full_query)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    topic = infer_topic(query)

    entry = {
        "time": timestamp,
        "query": query,
        "response": response,
        "topic": topic
    }

    st.session_state.chat_history.append(entry)
    save_chat_history(st.session_state.chat_history)

    st.markdown(f"### 🤖 {st.session_state.assistant_name}'s Response")
    stream_text(response)
