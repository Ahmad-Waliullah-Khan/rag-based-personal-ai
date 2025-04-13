import hashlib
import json
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

UPLOAD_DIR = "my_context/uploads"
VECTOR_DB_PATH = "db"
HASH_CACHE_FILE = "file_hash_cache.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_hash_cache():
    if os.path.exists(HASH_CACHE_FILE):
        return json.load(open(HASH_CACHE_FILE))
    return {}

def save_hash_cache(cache):
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_supported_docs():
    files = []
    for file_path in Path(UPLOAD_DIR).rglob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            files.append(file_path)
    return files

def embed_uploaded_documents():
    hash_cache = load_hash_cache()
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    updated = False

    for path in get_supported_docs():
        file_hash = get_file_hash(path)
        if hash_cache.get(str(path)) == file_hash:
            continue

        try:
            loader = (
                UnstructuredPDFLoader(str(path)) if path.suffix == ".pdf"
                else TextLoader(str(path))
            )
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = str(path.name)
            split = splitter.split_documents(loaded)
            docs.extend(split)
            hash_cache[str(path)] = file_hash
            updated = True
        except Exception as e:
            print(f"⚠️ Could not embed {path.name}: {e}")

    if docs:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
        vectordb.persist()
    save_hash_cache(hash_cache)
    return updated
