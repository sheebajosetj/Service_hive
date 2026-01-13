import os
from typing import TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

print("✅ new.py is running")

def load_vectorstore():
    kb_path = os.path.abspath("data/knowledge_base.md")
    print("\n🔍 Python is reading THIS file:")
    print(kb_path)

    print("📂 File exists:", os.path.exists(kb_path))
    print("📏 File size:", os.path.getsize(kb_path) if os.path.exists(kb_path) else "N/A")

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("📝 First 200 chars:", repr(text[:200]))
    print("📐 KB length:", len(text))

    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("📦 Chunks:", len(chunks))

    if not chunks:
        raise ValueError("❌ Knowledge base loaded but EMPTY")

    return Chroma.from_documents(chunks, embeddings)

if __name__ == "__main__":
    print("🚀 Calling load_vectorstore()")
    load_vectorstore()
