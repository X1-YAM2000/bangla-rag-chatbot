"""
api.py  -  FastAPI Backend for Bangla RAG Chatbot

Start the server:
    cd src && uvicorn api:app --reload --port 8000
    # or from project root:
    uvicorn src.api:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs    <- Swagger UI
    http://localhost:8000/redoc   <- ReDoc
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure rag_engine.py is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import BanglaRAGEngine

# ── Global engine instance ────────────────────────────────────────────────────
rag: Optional[BanglaRAGEngine] = None


# ── Lifespan context manager (FastAPI >= 0.93, replaces deprecated on_event) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    faq_hint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "faq_data.json"
    )
    rag = BanglaRAGEngine(faq_hint)
    print("RAG Engine ready.")
    yield  # server is running
    # shutdown cleanup (nothing needed)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bangla RAG Chatbot API",
    description=(
        "Bangla FAQ chatbot - TF-IDF retrieval + optional LLM refinement.\n\n"
        "Set BANGLA_LLM_PROVIDER=ollama or openai env var to enable answer refinement."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question:  str
    topic_id:  Optional[int]   = None   # None = search all topics
    threshold: Optional[float] = 0.1   # similarity cutoff (0-1)
    refine:    Optional[bool]  = True   # apply LLM refinement if configured


class FAQItem(BaseModel):
    id:         int
    question:   str
    answer:     str
    difficulty: str


class TopicItem(BaseModel):
    id:        int
    name:      str
    name_en:   str
    faq_count: int


class ChatResponse(BaseModel):
    found:            bool
    answer:           str
    raw_answer:       str
    similarity:       float
    topic:            Optional[str]
    difficulty:       Optional[str]
    matched_question: Optional[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Bangla RAG Chatbot API is running"}


@app.get("/topics", response_model=List[TopicItem], summary="List all topics")
def get_topics():
    """Return all 5 available topics with FAQ counts."""
    if rag is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    result = []
    for topic_id, name, name_en in rag.get_all_topics():
        result.append(TopicItem(
            id=topic_id,
            name=name,
            name_en=name_en,
            faq_count=len(rag.get_faqs_by_topic(topic_id)),
        ))
    return result


@app.get("/topics/{topic_id}/faqs", response_model=List[FAQItem],
         summary="All FAQs for a topic")
def get_faqs(topic_id: int):
    """Return every FAQ for the given topic_id (1-5)."""
    if rag is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    faqs = rag.get_faqs_by_topic(topic_id)
    if not faqs:
        raise HTTPException(status_code=404, detail=f"No FAQs for topic_id={topic_id}")
    return [
        FAQItem(
            id=f.get("id", i),
            question=f["question"],
            answer=f["answer"],
            difficulty=f.get("difficulty", "মাঝারি"),
        )
        for i, f in enumerate(faqs, 1)
    ]


@app.post("/chat", response_model=ChatResponse, summary="Ask a Bangla question")
def chat(req: QuestionRequest):
    """
    Ask a question in Bangla and receive the best-matching FAQ answer.

    - **question**  : Your Bangla question (required)
    - **topic_id**  : Restrict to a specific topic 1-5 (optional)
    - **threshold** : Minimum cosine similarity 0-1 (default 0.1)
    - **refine**    : Apply LLM refinement if BANGLA_LLM_PROVIDER is set (default true)
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = rag.search(
        query=req.question,
        topic_id=req.topic_id,
        threshold=req.threshold if req.threshold is not None else 0.1,
        refine=req.refine if req.refine is not None else True,
    )
    return ChatResponse(
        found=result["found"],
        answer=result["answer"],
        raw_answer=result.get("raw_answer", ""),
        similarity=result["similarity"],
        topic=result.get("topic"),
        difficulty=result.get("difficulty"),
        matched_question=result.get("matched_question"),
    )


@app.get("/search", response_model=ChatResponse, summary="Quick GET search")
def search_get(
    q:         str,
    topic_id:  Optional[int]   = None,
    threshold: Optional[float] = 0.1,
):
    """
    Convenience GET endpoint for browser / curl testing.

    Example:
        /search?q=পানি+কত+লিটার&topic_id=2
    """
    return chat(QuestionRequest(question=q, topic_id=topic_id, threshold=threshold))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
