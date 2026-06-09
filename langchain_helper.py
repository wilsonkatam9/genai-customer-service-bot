import os
import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# Sentence transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

FAISS_INDEX_PATH = "faiss_index/index.faiss"
FAISS_DATA_PATH = "faiss_index/index.pkl"
CSV_PATH = "dataset/dataset.csv"

def load_dataset():
    return pd.read_csv(CSV_PATH, encoding="latin-1")

def build_vector_db(df=None):
    """Build FAISS vector database from dataset"""
    os.makedirs("faiss_index", exist_ok=True)
    
    if df is None:
        df = load_dataset()
    
    # Create documents list
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "prompt": str(row["prompt"]),
            "response": str(row["response"])
        })
    
    # Generate embeddings
    texts = [doc["prompt"] for doc in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and documents
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_DATA_PATH, "wb") as f:
        pickle.dump(documents, f)
    
    print(f"✅ Vector DB built with {len(documents)} entries")
    return index, documents

def load_vector_db():
    """Load existing FAISS index"""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_DATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_DATA_PATH, "rb") as f:
            documents = pickle.load(f)
        return index, documents
    else:
        # Build if not exists
        df = load_dataset()
        return build_vector_db(df)

def retrieve_relevant(question, index, documents, top_k=5):
    """Retrieve top-k relevant Q&A pairs using FAISS"""
    q_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):
            results.append({
                "prompt": documents[idx]["prompt"],
                "response": documents[idx]["response"],
                "score": float(distances[0][i])
            })
    return results

def add_to_knowledge_base(new_prompt, new_response):
    """Dynamically add new entry to CSV and rebuild FAISS index"""
    df = load_dataset()
    
    # Add new row
    new_row = pd.DataFrame([[new_prompt, new_response]], columns=["prompt", "response"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="latin-1")
    
    # Rebuild vector DB
    build_vector_db(df)
    print(f"✅ Added new entry and rebuilt vector DB")
    return len(df)

def get_answer(question):
    """Get answer using FAISS retrieval + Gemini"""
    index, documents = load_vector_db()
    relevant = retrieve_relevant(question, index, documents)
    
    context = ""
    for r in relevant:
        context += f"Q: {r['prompt']}\nA: {r['response']}\n\n"
    
    prompt = f"""You are a helpful customer service assistant for an e-learning company.
Use the knowledge base below to answer the question.

KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

Instructions:
- If the question matches knowledge base, use that answer
- If not, use your general knowledge
- Be friendly and detailed

ANSWER:"""
    
    response = model.generate_content(prompt)
    return response.text

def create_vector_db():
    """Called on startup to ensure vector DB exists"""
    build_vector_db()

def get_qa_chain():
    return None