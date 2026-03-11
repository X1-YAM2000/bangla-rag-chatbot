"""
demo_chatbot.py  —  Non-interactive demo of the Bangla RAG Chatbot

Bug fixed: faq_data.json path now resolved by rag_engine automatically.
Run from any directory:  python src/demo_chatbot.py
                      OR cd src && python demo_chatbot.py
"""

import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import BanglaRAGEngine


# ── Demo scenarios ────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "title":   "Education / শিক্ষা",
        "topic_id": 1,
        "queries": [
            "একটি ভালো স্কুল কিভাবে খুজে পাব?",
            "অনলাইনে পড়াশোনা কার্যকর?",
        ],
    },
    {
        "title":   "Health / স্বাস্থ্য",
        "topic_id": 2,
        "queries": [
            "দিনে কত লিটার পানি পানো উচিত?",
            "সুস্থ থাকার উপায় কী?",
        ],
    },
    {
        "title":   "Travel / ভ্রমণ",
        "topic_id": 3,
        "queries": [
            "বাংলাদেশে ভ্রমণের সেরা সময় কখন?",
            "ভ্রমণ বাজেট কীভাবে পরিকল্পনা করব?",
        ],
    },
    {
        "title":   "Technology / প্রযুক্তি",
        "topic_id": 4,
        "queries": [
            "AI কী এবং এটি কীভাবে কাজ করে?",
            "সাইবার নিরাপত্তা কেন জরুরি?",
        ],
    },
    {
        "title":   "Sports / খেলাধুলা",
        "topic_id": 5,
        "queries": [
            "নিয়মিত খেলাধুলার উপকার কী?",
            "ফুটবল খেলা শিখতে চাই",
        ],
    },
]


def run_demo(delay: float = 0.4) -> bool:
    print("\n" + "=" * 70)
    print("  Bangla RAG Chatbot — DEMO")
    print("  বাংলা RAG চ্যাটবট — ডেমো")
    print("=" * 70 + "\n")

    # Resolve FAQ path the same way the chatbot does
    faq_hint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "faq_data.json"
    )

    try:
        rag = BanglaRAGEngine(faq_hint)
        print("Chatbot ready!\n")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False

    total     = 0
    answered  = 0
    scores    = []

    for scenario in DEMO_SCENARIOS:
        print("\n" + "=" * 70)
        print(f"  {scenario['title']}")
        print("=" * 70)

        for query in scenario["queries"]:
            total += 1
            print(f"\nUser : {query}")
            time.sleep(delay)

            result = rag.search(query, scenario["topic_id"])

            print("Bot  :", end=" ")
            if result["found"]:
                answered += 1
                scores.append(result["similarity"])
                print(f"[Found | {result['similarity']:.0%} match]")
                print(f"       {result['answer']}")
                print(f"       Topic: {result['topic']}  |  "
                      f"Difficulty: {result.get('difficulty','N/A')}")
            else:
                print("[Not found]")
                print(f"       {result['answer']}")

            print("-" * 70)
            time.sleep(delay)

    # Summary
    print("\n" + "=" * 70)
    print("Demo complete!")
    print(f"  Answered   : {answered}/{total} ({answered/total:.0%})")
    if scores:
        print(f"  Avg score  : {sum(scores)/len(scores):.1%}")
        print(f"  Best score : {max(scores):.1%}")
    print("=" * 70)
    print("\nTo run the full interactive chatbot:")
    print("  python src/bangla_rag_chatbot.py")
    print("\nTo start the FastAPI server:")
    print("  uvicorn src.api:app --reload --port 8000\n")

    return True


if __name__ == "__main__":
    ok = run_demo()
    sys.exit(0 if ok else 1)
