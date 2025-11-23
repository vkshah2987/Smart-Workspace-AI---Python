import os, uuid
import pdfplumber
from docx import Document
import pandas as pd

CHUNK_SIZE = 500
CHUNK_STRIDE = 100

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        return "\n".join(texts)
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
        
def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i+CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += CHUNK_SIZE - CHUNK_STRIDE
    return chunks

def process_document(path, doc_id):
    text = extract_text(path)
    raw_chunks = chunk_text(text)
    ret = []
    for seq, txt in enumerate(raw_chunks):
        ret.append({
            "chunk_id": f"{doc_id}__{seq}",
            "text": txt,
            "seq": seq,
            "tokens": len(txt.split())
        })
    return ret