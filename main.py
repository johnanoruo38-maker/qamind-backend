"""
NLP Question Answering API — FastAPI Backend
Supports: TF-IDF retrieval-based QA + optional Claude API fallback
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import json
import os
import re
import math
import string
from collections import Counter
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="NLP QA API",
    description="Natural Language Question Answering using TF-IDF + cosine similarity",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data Layer ────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = os.getenv("KB_PATH", "knowledge_base.json")

def load_knowledge_base() -> list[dict]:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_base: list[dict] = load_knowledge_base()


# ── NLP Utilities ─────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and",
    "or","but","not","with","this","that","are","was","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","i","you",
    "he","she","we","they","what","how","why","when","where","which",
    "who","whom","whose","about","from","by","as","so","if","then",
}

def preprocess(text: str) -> list[str]:
    """Lowercase, remove punctuation, tokenize, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    count = Counter(tokens)
    total = len(tokens) or 1
    return {term: freq / total for term, freq in count.items()}


def compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    idf: dict[str, float] = {}
    all_terms = set(t for doc in corpus for t in doc)
    for term in all_terms:
        df = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Pre-build TF-IDF index ────────────────────────────────────────────────────
_corpus_tokens = [preprocess(item["question"]) for item in knowledge_base]
_idf = compute_idf(_corpus_tokens)
_doc_vectors = [tfidf_vector(tokens, _idf) for tokens in _corpus_tokens]


def find_best_answer(question: str, top_k: int = 1) -> list[dict]:
    q_tokens = preprocess(question)
    q_vec = tfidf_vector(q_tokens, _idf)

    scored = []
    for idx, doc_vec in enumerate(_doc_vectors):
        score = cosine_similarity(q_vec, doc_vec)
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        results.append({
            "question": knowledge_base[idx]["question"],
            "answer": knowledge_base[idx]["answer"],
            "category": knowledge_base[idx].get("category", "general"),
            "confidence": round(score, 4),
        })
    return results


# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 1

    @validator("question")
    def question_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 500:
            raise ValueError("Question must be ≤ 500 characters")
        return v

    @validator("top_k")
    def top_k_range(cls, v):
        if v is not None and not (1 <= v <= 5):
            raise ValueError("top_k must be between 1 and 5")
        return v


class AskResponse(BaseModel):
    answer: str
    confidence: float
    matched_question: str
    category: str
    source: str = "knowledge_base"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "kb_size": len(knowledge_base)}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy", "kb_entries": len(knowledge_base)}


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(payload: AskRequest):
    """
    Accept a natural-language question and return the best-matching answer
    from the knowledge base using TF-IDF cosine similarity.
    """
    results = find_best_answer(payload.question, top_k=payload.top_k or 1)

    if not results or results[0]["confidence"] < 0.05:
        raise HTTPException(
            status_code=404,
            detail="No confident answer found. Try rephrasing your question.",
        )

    best = results[0]
    return AskResponse(
        answer=best["answer"],
        confidence=best["confidence"],
        matched_question=best["question"],
        category=best["category"],
        source="knowledge_base",
    )


@app.get("/questions", tags=["kb"])
def list_questions(category: Optional[str] = None):
    """Return all questions in the knowledge base (optionally filtered by category)."""
    data = knowledge_base
    if category:
        data = [item for item in data if item.get("category") == category]
    return {"count": len(data), "questions": [item["question"] for item in data]}