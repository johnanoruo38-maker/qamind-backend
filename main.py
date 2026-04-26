from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import json, os, math, string, httpx, re
from collections import Counter
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NLP QA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE_PATH = os.getenv("KB_PATH", "knowledge_base.json")

def load_knowledge_base():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_base = load_knowledge_base()

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and",
    "or","but","not","with","this","that","are","was","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","i","you",
    "he","she","we","they","what","how","why","when","where","which",
    "who","whom","whose","about","from","by","as","so","if","then",
}

def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

def compute_tf(tokens):
    count = Counter(tokens)
    total = len(tokens) or 1
    return {term: freq / total for term, freq in count.items()}

def compute_idf(corpus):
    N = len(corpus)
    idf = {}
    all_terms = set(t for doc in corpus for t in doc)
    for term in all_terms:
        df = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf

def tfidf_vector(tokens, idf):
    tf = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}

def cosine_similarity(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

_corpus_tokens = [preprocess(item["question"]) for item in knowledge_base]
_idf = compute_idf(_corpus_tokens)
_doc_vectors = [tfidf_vector(tokens, _idf) for tokens in _corpus_tokens]

def find_best_answer(question, top_k=1):
    q_tokens = preprocess(question)
    q_vec = tfidf_vector(q_tokens, _idf)
    scored = sorted(
        [(cosine_similarity(q_vec, doc_vec), idx)
         for idx, doc_vec in enumerate(_doc_vectors)],
        reverse=True
    )
    return [{
        "question": knowledge_base[idx]["question"],
        "answer": knowledge_base[idx]["answer"],
        "category": knowledge_base[idx].get("category", "general"),
        "confidence": round(score, 4),
    } for score, idx in scored[:top_k]]


def extract_topic(question: str) -> str:
    """Extract the core topic from a question."""
    q = question.lower().strip()
    patterns = [
        r"^what is (a |an |the )?",
        r"^what are (a |an |the )?",
        r"^who is (a |an |the )?",
        r"^who was (a |an |the )?",
        r"^how does (a |an |the )?",
        r"^how do (a |an |the )?",
        r"^where is (a |an |the )?",
        r"^when (was|is|did) (a |an |the )?",
        r"^define (a |an |the )?",
        r"^tell me about (a |an |the )?",
        r"^explain (a |an |the )?",
        r"^describe (a |an |the )?",
    ]
    for pattern in patterns:
        q = re.sub(pattern, "", q)
    return q.strip().rstrip("?").strip()


def clean_answer(text: str, max_sentences: int = 3) -> str:
    """Return a clean short answer from a long text."""
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    result = '. '.join(sentences[:max_sentences])
    if result and not result.endswith('.'):
        result += '.'
    return result


async def search_wikipedia(topic: str, client: httpx.AsyncClient) -> str | None:
    """Use Wikipedia opensearch to find best matching article title."""
    try:
        # Step 1: Use opensearch to find the best matching title
        search_res = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": topic,
                "limit": "5",
                "format": "json",
            },
            timeout=8.0,
        )
        data = search_res.json()
        titles = data[1] if len(data) > 1 else []

        if not titles:
            return None

        # Step 2: Try each title until we get a good summary
        for title in titles:
            summary_res = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                timeout=8.0,
                follow_redirects=True,
            )
            if summary_res.status_code != 200:
                continue

            page = summary_res.json()

            # Skip disambiguation pages
            if page.get("type") == "disambiguation":
                continue

            extract = page.get("extract", "")
            if extract and len(extract) > 40:
                return clean_answer(extract)

    except Exception as e:
        print(f"Wikipedia opensearch error: {e}")
    return None


async def ask_wikipedia(question: str) -> dict | None:
    """Search Wikipedia for any question — completely free."""
    try:
        async with httpx.AsyncClient() as client:
            topic = extract_topic(question)

            # Try with extracted topic first
            answer = await search_wikipedia(topic, client)

            # If that fails try the full question
            if not answer:
                answer = await search_wikipedia(question, client)

            if answer:
                return {
                    "answer": answer,
                    "confidence": 0.68,
                    "matched_question": question,
                    "category": "general",
                    "source": "wikipedia",
                }
    except Exception as e:
        print(f"Wikipedia error: {e}")
    return None


async def ask_duckduckgo(question: str) -> dict | None:
    """DuckDuckGo Instant Answer API — free backup."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": question,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                timeout=8.0,
            )
            data = response.json()
            answer = (
                data.get("AbstractText") or
                data.get("Answer") or
                data.get("Definition") or ""
            )
            if answer and len(answer) > 30:
                return {
                    "answer": answer,
                    "confidence": 0.65,
                    "matched_question": question,
                    "category": "general",
                    "source": "duckduckgo",
                }
    except Exception as e:
        print(f"DuckDuckGo error: {e}")
    return None


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 1

    @validator("question")
    def question_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or less")
        return v


class AskResponse(BaseModel):
    answer: str
    confidence: float
    matched_question: str
    category: str
    source: str = "knowledge_base"


@app.get("/")
def root():
    return {"status": "ok", "kb_size": len(knowledge_base)}

@app.get("/health")
def health():
    return {"status": "healthy", "kb_entries": len(knowledge_base)}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    # Step 1 — Knowledge base (instant, always free)
    results = find_best_answer(payload.question)
    if results and results[0]["confidence"] >= 0.15:
        best = results[0]
        return AskResponse(
            answer=best["answer"],
            confidence=best["confidence"],
            matched_question=best["question"],
            category=best["category"],
            source="knowledge_base",
        )

    # Step 2 — Wikipedia opensearch (handles everyday words)
    wiki_result = await ask_wikipedia(payload.question)
    if wiki_result:
        return AskResponse(**wiki_result)

    # Step 3 — Nothing found
    raise HTTPException(
        status_code=404,
        detail="No answer found. Try rephrasing your question.",
    )


@app.get("/questions")
def list_questions(category: Optional[str] = None):
    data = knowledge_base
    if category:
        data = [item for item in data if item.get("category") == category]
    return {"count": len(data), "questions": [item["question"] for item in data]}