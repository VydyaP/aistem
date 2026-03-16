import os
from typing import List
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
import json

def call_granite_llm(prompt, model="granite3.3:8b", api_url="http://localhost:11434/api/generate"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(api_url, json=payload, stream=True)
    response.raise_for_status()
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            obj = json.loads(data)
            full_response += obj.get("response", "")
            if obj.get("done", False):
                break
    return full_response

class RAGRetriever:
    def __init__(self, docs_dir: str = "ai_stem_tutor/source_docs", index_dir: str = "ai_stem_tutor/vector_store"):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_pdfs(self) -> List[str]:
        texts = []
        for filename in os.listdir(self.docs_dir):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.docs_dir, filename)
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    texts.append(full_text)
        return texts

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 5,
        max_chars: int = 450,
        overlap_sentences: int = 1,
        use_smart_chunking: bool = True,
    ) -> List[str]:
        """Chunk by sentences. Optional: cap by char length and add overlap for better retrieval."""
        sentences = sent_tokenize(text)
        if not sentences:
            return [text] if text.strip() else []

        if not use_smart_chunking:
            # Original: fixed number of sentences, no overlap
            return [
                " ".join(sentences[i : i + chunk_size])
                for i in range(0, len(sentences), chunk_size)
            ]

        # Smart: respect max_chars and add overlap; keep sentence boundaries
        chunks = []
        i = 0
        while i < len(sentences):
            group = []
            length = 0
            j = i
            while j < len(sentences) and (length + len(sentences[j]) + 1 <= max_chars or not group):
                group.append(sentences[j])
                length += len(sentences[j]) + 1
                j += 1
                if len(group) >= chunk_size:
                    break
            if group:
                chunks.append(" ".join(group))
            # Next start: step back by overlap so chunks share context
            next_i = j - overlap_sentences if overlap_sentences > 0 and j < len(sentences) else j
            i = max(i + 1, next_i)  # always advance at least one sentence
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List:
        embeddings = self.embedding_model.encode(chunks)
        return embeddings

    def build_faiss_index(self, embeddings: List, chunks: List[str]):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(self.index_dir, "faiss_index.bin"))
        id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
        with open(os.path.join(self.index_dir, "id_to_chunk.pkl"), "wb") as f:
            pickle.dump(id_to_chunk, f)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self.embedding_model.encode([query]).astype('float32')
        index = faiss.read_index(os.path.join(self.index_dir, "faiss_index.bin"))
        D, I = index.search(query_emb, top_k)
        with open(os.path.join(self.index_dir, "id_to_chunk.pkl"), "rb") as f:
            id_to_chunk = pickle.load(f)
        return [id_to_chunk[i] for i in I[0] if i in id_to_chunk]

# (Keep your call_granite_llm function as is)
if __name__ == "__main__":
    import nltk
    nltk.download("punkt")
    retriever = RAGRetriever()
    print("Extracting text from PDFs...")
    texts = retriever.extract_text_from_pdfs()
    print(f"Extracted {len(texts)} documents.")
    all_chunks = []
    for text in texts:
        chunks = retriever.chunk_text(text)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    print("Generating embeddings...")
    embeddings = retriever.generate_embeddings(all_chunks)
    print("Building FAISS index...")
    retriever.build_faiss_index(embeddings, all_chunks)
    print("Index built!")