"""
rag_engine.py  —  Bangla RAG Engine (v2)

Retrieval  : TF-IDF + Cosine Similarity (sklearn — no internet needed)
Generation : Optional LLM refinement via Ollama or OpenAI-compatible API
             Set BANGLA_LLM_PROVIDER env var to enable (see README).
             If unset the engine returns the raw FAQ answer — fully offline.
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional


# ── LLM refinement config (all optional) ─────────────────────────────────────
#
#   Ollama (free, local):
#       BANGLA_LLM_PROVIDER = ollama
#       BANGLA_LLM_MODEL    = llama3          (or mistral, phi3, gemma ...)
#       BANGLA_LLM_BASE_URL = http://localhost:11434/v1   <- default
#
#   OpenAI / compatible cloud:
#       BANGLA_LLM_PROVIDER = openai
#       BANGLA_LLM_MODEL    = gpt-4o-mini
#       OPENAI_API_KEY      = sk-...
#
#   Leave BANGLA_LLM_PROVIDER empty to skip LLM entirely (offline mode).
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("BANGLA_LLM_PROVIDER", "").lower()
LLM_MODEL      = os.getenv("BANGLA_LLM_MODEL", "llama3")
LLM_BASE_URL   = os.getenv("BANGLA_LLM_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_refinement_prompt(user_question: str, faq_answer: str, topic: str):
    """Return (system, user) prompt pair for Bangla answer refinement."""
    system = (
        "তুমি একজন বাংলা ভাষা বিশেষজ্ঞ। "
        "তোমাকে একটি প্রশ্ন এবং একটি সংক্ষিপ্ত FAQ উত্তর দেওয়া হবে। "
        "FAQ উত্তরটিকে আরও স্বাভাবিক, বিস্তারিত ও প্রাঞ্জল বাংলায় পুনর্লিখন করো। "
        "তথ্য পরিবর্তন করো না - শুধু ভাষা পরিশীলন করো। "
        "উত্তর সরাসরি দাও, অতিরিক্ত ভূমিকা ছাড়া।"
    )
    user = (
        f"বিষয়: {topic}\n\n"
        f"প্রশ্ন: {user_question}\n\n"
        f"FAQ উত্তর:\n{faq_answer}\n\n"
        "পরিশীলিত উত্তর লিখো:"
    )
    return system, user


def _call_llm(system: str, user_msg: str, base_url: str, model: str, api_key: str) -> Optional[str]:
    """Call any OpenAI-compatible /v1/chat/completions endpoint."""
    try:
        from openai import OpenAI  # pip install openai
        client = OpenAI(base_url=base_url, api_key=api_key or "ollama")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        print("  'openai' package not installed. Run: pip install openai")
    except Exception as exc:
        print(f"  LLM refinement failed: {exc}")
    return None


def refine_answer(user_question: str, raw_answer: str, topic: str) -> str:
    """
    Rewrite raw_answer in more fluent Bangla using the configured LLM.
    Falls back to raw_answer if no LLM is configured or the call fails.
    """
    if not LLM_PROVIDER:
        return raw_answer

    system, user_msg = _build_refinement_prompt(user_question, raw_answer, topic)

    if LLM_PROVIDER == "ollama":
        refined = _call_llm(system, user_msg, LLM_BASE_URL, LLM_MODEL, "ollama")
    elif LLM_PROVIDER == "openai":
        refined = _call_llm(system, user_msg, "https://api.openai.com/v1", LLM_MODEL, OPENAI_API_KEY)
    else:
        print(f"  Unknown BANGLA_LLM_PROVIDER='{LLM_PROVIDER}'. Using raw answer.")
        refined = None

    return refined if refined else raw_answer


# ── Path resolver ─────────────────────────────────────────────────────────────

def _resolve_faq_path(faq_file_path: str) -> str:
    """
    Accept the path as-is if it exists, otherwise try common relative locations
    so scripts work when run from project root OR from src/.
    """
    if os.path.exists(faq_file_path):
        return faq_file_path

    src_dir     = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)

    candidates = [
        os.path.join(src_dir, faq_file_path),
        os.path.join(src_dir, "..", "data", "faq_data.json"),
        os.path.join(src_dir, "faq_data.json"),
        os.path.join(project_dir, "data", "faq_data.json"),
        os.path.join(project_dir, "faq_data.json"),
        os.path.join(os.getcwd(), "data", "faq_data.json"),
        os.path.join(os.getcwd(), "faq_data.json"),
    ]
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.exists(norm):
            return norm

    tried = "\n".join(f"  {os.path.normpath(c)}" for c in candidates)
    raise FileNotFoundError(
        f"faq_data.json not found. Tried:\n{tried}"
    )


# ── Main RAG Engine ───────────────────────────────────────────────────────────

class BanglaRAGEngine:
    """
    Retrieval-Augmented Generation engine for Bangla FAQ answering.

    Retrieval : TF-IDF (char n-grams) + cosine similarity - no GPU, no internet.
    Generation: Optional LLM refinement (Ollama / OpenAI) via env vars.
    """

    def __init__(self, faq_file_path: str):
        self.faq_file_path = _resolve_faq_path(faq_file_path)
        self.faq_data      = self._load_faq_data()

        self.all_questions: List[str]  = []
        self.all_answers:   List[Dict] = []
        self.all_topics:    List[str]  = []
        self.topic_index:   Dict[int, str] = {}

        # TF-IDF with character n-grams - works well for Bangla script
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )
        self.faq_vectors = None
        self._build_index()

        if LLM_PROVIDER:
            print(f"LLM refinement enabled: {LLM_PROVIDER} / {LLM_MODEL}")
        else:
            print("Offline mode (TF-IDF). Set BANGLA_LLM_PROVIDER=ollama to enable LLM.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_faq_data(self) -> Dict:
        with open(self.faq_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        for topic in self.faq_data.get("topics", []):
            tid  = topic["id"]
            name = topic["name"]
            self.topic_index[tid] = name
            for faq in topic.get("faqs", []):
                self.all_questions.append(faq["question"])
                self.all_answers.append({
                    "answer":     faq["answer"],
                    "topic":      name,
                    "difficulty": faq.get("difficulty", "মাঝারি"),
                    "question":   faq["question"],
                })
                self.all_topics.append(name)

        if self.all_questions:
            print(f"Building TF-IDF index for {len(self.all_questions)} FAQs...")
            self.faq_vectors = self.vectorizer.fit_transform(self.all_questions)
            print("Index built successfully!")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all_topics(self) -> List[Tuple[int, str, str]]:
        """Return list of (id, bangla_name, english_name) tuples."""
        return [
            (t["id"], t["name"], t["name_en"])
            for t in self.faq_data.get("topics", [])
        ]

    def get_faqs_by_topic(self, topic_id: int) -> List[Dict]:
        """Return all FAQ dicts for the given topic_id."""
        for topic in self.faq_data.get("topics", []):
            if topic["id"] == topic_id:
                return topic.get("faqs", [])
        return []

    def search(
        self,
        query: str,
        topic_id: Optional[int] = None,
        threshold: float = 0.1,
        refine: bool = True,
    ) -> Dict:
        """
        Find the best-matching FAQ answer for *query*.

        Args:
            query     : User question in Bangla (or English).
            topic_id  : Restrict search to this topic (None = all topics).
            threshold : Minimum cosine similarity to accept a match (0-1).
                        Default 0.1 - works well for TF-IDF on short Bangla text.
            refine    : Apply LLM refinement if configured.

        Returns a dict with keys:
            found, answer, raw_answer, similarity, topic, difficulty,
            matched_question
        """
        if not self.all_questions or self.faq_vectors is None:
            return {
                "found":            False,
                "answer":           "এখানে কোনো উত্তর খুঁজে পাওয়া যায়নি।",
                "raw_answer":       "",
                "similarity":       0.0,
                "topic":            None,
                "difficulty":       None,
                "matched_question": None,
            }

        query_vec    = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.faq_vectors)[0]

        # Zero-out scores outside the requested topic
        if topic_id is not None:
            topic_name = self.topic_index.get(topic_id)
            if topic_name:
                for i, t in enumerate(self.all_topics):
                    if t != topic_name:
                        similarities[i] = 0.0

        best_idx        = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        if best_similarity >= threshold:
            meta  = self.all_answers[best_idx]
            raw   = meta["answer"]
            final = refine_answer(query, raw, meta["topic"]) if refine else raw
            return {
                "found":            True,
                "answer":           final,
                "raw_answer":       raw,
                "similarity":       best_similarity,
                "topic":            meta["topic"],
                "difficulty":       meta["difficulty"],
                "matched_question": meta["question"],
            }

        return {
            "found":            False,
            "answer":           (
                "আমি এই প্রশ্নের সঠিক উত্তর খুঁজে পাই নি। "
                "দয়া করে আপনার প্রশ্নটি পরিষ্কার করুন "
                "বা একটি নির্দিষ্ট বিষয় নির্বাচন করুন।"
            ),
            "raw_answer":       "",
            "similarity":       best_similarity,
            "topic":            None,
            "difficulty":       None,
            "matched_question": None,
        }
