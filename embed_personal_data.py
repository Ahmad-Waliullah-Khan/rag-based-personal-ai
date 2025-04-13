from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredCSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib
import json
import os

# Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db"
HASH_CACHE_PATH = "embedded_hashes.json"
MAX_FILE_SIZE_MB = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Supported file loaders
EXTENSION_LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": UnstructuredPDFLoader,
    ".csv": UnstructuredCSVLoader,
}

def tag_from_path(path: Path) -> str:
    if "my_context" in path.parts:
        idx = path.parts.index("my_context")
        if len(path.parts) > idx + 1:
            return path.parts[idx + 1]
    return "general"

def get_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_hash_cache():
    if os.path.exists(HASH_CACHE_PATH):
        with open(HASH_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_hash_cache(cache):
    with open(HASH_CACHE_PATH, "w") as f:
        json.dump(cache, f)

def load_all_files(root_folder="my_context"):
    docs = []
    hash_cache = load_hash_cache()
    updated_cache = hash_cache.copy()

    for file_path in Path(root_folder).rglob("*"):
        if file_path.is_file():
            if file_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                print(f"⚠️ Skipping large file: {file_path.name}")
                continue

            ext = file_path.suffix.lower()
            loader_cls = EXTENSION_LOADERS.get(ext)
            if not loader_cls:
                continue

            file_hash = get_file_hash(file_path)
            if hash_cache.get(str(file_path)) == file_hash:
                print(f"⏭️ Skipping unchanged file: {file_path.name}")
                continue

            try:
                loader = loader_cls(str(file_path))
                loaded_docs = loader.load()
                tag = tag_from_path(file_path)
                for doc in loaded_docs:
                    doc.metadata["tag"] = tag
                    doc.metadata["source"] = str(file_path)
                docs.extend(loaded_docs)
                updated_cache[str(file_path)] = file_hash
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

    save_hash_cache(updated_cache)
    return docs

# Load, split, and embed
docs = load_all_files()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
split_docs = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
vectordb.persist()

print(f"✅ Embedded {len(split_docs)} document chunks with tags and metadata.")
