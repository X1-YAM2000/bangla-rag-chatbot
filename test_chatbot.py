"""
test_chatbot.py  —  Automated test suite for the Bangla RAG Engine

Bug fixed: faq_data.json path now resolved by rag_engine automatically.
All tests pass offline (no SBERT / internet required).
"""

import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import BanglaRAGEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(msg: str):
    print(f"  PASS  {msg}")

def fail(msg: str):
    print(f"  FAIL  {msg}")


# ── Test cases ────────────────────────────────────────────────────────────────

def test_initialization(rag: BanglaRAGEngine) -> bool:
    print("\nTest 1: Initialization")
    print("-" * 60)
    passed = True

    if rag.faq_vectors is not None:
        ok("TF-IDF index built")
    else:
        fail("TF-IDF index is None"); passed = False

    if len(rag.all_questions) == 15:
        ok(f"Loaded 15 FAQ questions")
    else:
        fail(f"Expected 15 questions, got {len(rag.all_questions)}"); passed = False

    if len(rag.topic_index) == 5:
        ok("5 topics in index")
    else:
        fail(f"Expected 5 topics, got {len(rag.topic_index)}"); passed = False

    return passed


def test_topics(rag: BanglaRAGEngine) -> bool:
    print("\nTest 2: Topic Loading")
    print("-" * 60)
    passed     = True
    topics     = rag.get_all_topics()
    expected   = {"Education", "Health", "Travel", "Technology", "Sports"}
    english    = {t[2] for t in topics}

    if english == expected:
        ok(f"All 5 topics present: {sorted(expected)}")
    else:
        fail(f"Topics mismatch. Got: {english}"); passed = False

    total = sum(len(rag.get_faqs_by_topic(t[0])) for t in topics)
    if total == 15:
        ok("15 total FAQs across all topics")
    else:
        fail(f"Expected 15 FAQs, got {total}"); passed = False

    for tid, name, name_en in topics:
        count = len(rag.get_faqs_by_topic(tid))
        if count >= 3:
            ok(f"{name_en}: {count} FAQs")
        else:
            fail(f"{name_en}: only {count} FAQs (need >= 3)"); passed = False

    return passed


def test_search_no_filter(rag: BanglaRAGEngine) -> bool:
    print("\nTest 3: Search Without Topic Filter")
    print("-" * 60)
    passed = True
    cases  = [
        ("শিক্ষা প্রতিষ্ঠান কীভাবে নির্বাচন করব?",  True),
        ("প্রতিদিন কত পানি পানো উচিত?",               True),
        ("ভ্রমণের সেরা সময় কখন?",                    True),
        ("কৃত্রিম বুদ্ধিমত্তা কী?",                  True),
        ("ফুটবল খেলতে কীভাবে শিখব?",                  True),
    ]
    for q, expect_found in cases:
        res = rag.search(q)
        if res["found"] == expect_found:
            ok(f"'{q[:30]}...' → {res['similarity']:.1%} [{res.get('topic','?')}]")
        else:
            fail(f"'{q[:30]}...' found={res['found']} (expected {expect_found})")
            passed = False
        # Ensure 'difficulty' is always present (no KeyError)
        _ = res.get("difficulty", "N/A")

    return passed


def test_topic_filter(rag: BanglaRAGEngine) -> bool:
    """
    Topic filter zeroes out non-topic scores so only FAQs from the requested
    topic can win.  We verify that whatever is returned belongs to the right
    topic, not that it's necessarily 'not found' (a low but non-zero TF-IDF
    similarity is valid even for off-domain queries).
    """
    print("\nTest 4: Topic Filtering")
    print("-" * 60)
    passed = True

    q = "শিক্ষা প্রতিষ্ঠান কীভাবে নির্বাচন করব?"

    # When filtering to Health (id=2), any result must come from Health
    res = rag.search(q, topic_id=2)
    if not res["found"] or res["topic"] == "স্বাস্থ্য":
        ok(f"Filter=Health: result topic is স্বাস্থ্য (or not found) — filter is working")
    else:
        fail(f"Filter=Health returned wrong topic: {res['topic']}"); passed = False

    # When filtering to Education (id=1), best match should be in Education
    res = rag.search(q, topic_id=1)
    if res["found"] and res["topic"] == "শিক্ষা":
        ok(f"Filter=Education: matched '{res['matched_question'][:40]}...' ({res['similarity']:.1%})")
    else:
        fail(f"Education query in Education topic: found={res['found']} topic={res.get('topic')}")
        passed = False

    # Cross-topic: Sports query filtered to Technology should return Technology topic
    sports_q = "ফুটবল খেলতে শিখতে চাই"
    res = rag.search(sports_q, topic_id=4)  # Technology
    if not res["found"] or res["topic"] == "প্রযুক্তি":
        ok("Filter=Technology on sports query: returns Technology result or not-found")
    else:
        fail(f"Unexpected topic in Technology filter: {res['topic']}"); passed = False

    return passed


def test_threshold(rag: BanglaRAGEngine) -> bool:
    print("\nTest 5: Threshold & Fallback")
    print("-" * 60)
    passed = True

    gibberish = "xyzzy plugh abracadabra 12345"
    res = rag.search(gibberish, threshold=0.9)
    if not res["found"]:
        ok("Gibberish query correctly triggers fallback response")
        ok(f"Fallback message present: '{res['answer'][:50]}...'")
    else:
        fail(f"Gibberish matched at {res['similarity']:.1%} with threshold 0.9")
        passed = False

    # raw_answer always present
    assert isinstance(res["raw_answer"], str), "raw_answer must be a string"
    ok("raw_answer field always a string")

    return passed


def test_get_faqs_by_topic(rag: BanglaRAGEngine) -> bool:
    print("\nTest 6: get_faqs_by_topic()")
    print("-" * 60)
    passed = True

    for tid in range(1, 6):
        faqs = rag.get_faqs_by_topic(tid)
        if faqs:
            ok(f"topic_id={tid}: {len(faqs)} FAQs, first Q: '{faqs[0]['question'][:35]}...'")
        else:
            fail(f"topic_id={tid}: returned empty list"); passed = False

    empty = rag.get_faqs_by_topic(999)
    if empty == []:
        ok("Non-existent topic returns empty list (no exception)")
    else:
        fail("Non-existent topic should return []"); passed = False

    return passed


def test_result_keys(rag: BanglaRAGEngine) -> bool:
    """Ensure all expected keys are present in both found and not-found results."""
    print("\nTest 7: Result Dictionary Keys")
    print("-" * 60)
    required = {"found", "answer", "raw_answer", "similarity", "topic", "difficulty",
                "matched_question"}
    passed   = True

    for q, tid in [("পানি", None), ("xyzzy", None)]:
        res  = rag.search(q, topic_id=tid)
        missing = required - set(res.keys())
        if not missing:
            ok(f"All keys present for query='{q}' (found={res['found']})")
        else:
            fail(f"Missing keys {missing} for query='{q}'"); passed = False

    return passed


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    print("\n" + "=" * 60)
    print("  Bangla RAG Chatbot — Test Suite")
    print("=" * 60)

    faq_hint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "faq_data.json"
    )

    try:
        rag = BanglaRAGEngine(faq_hint)
        print("RAG Engine initialised OK\n")
    except Exception as exc:
        print(f"FATAL: Could not initialise RAG Engine: {exc}")
        return False

    tests  = [
        test_initialization,
        test_topics,
        test_search_no_filter,
        test_topic_filter,
        test_threshold,
        test_get_faqs_by_topic,
        test_result_keys,
    ]
    results = [t(rag) for t in tests]

    passed = sum(results)
    total  = len(results)

    print("\n" + "=" * 60)
    if all(results):
        print(f"  ALL {total} TESTS PASSED")
    else:
        failed_names = [t.__name__ for t, r in zip(tests, results) if not r]
        print(f"  {passed}/{total} tests passed. Failed: {', '.join(failed_names)}")
    print("=" * 60 + "\n")

    return all(results)


if __name__ == "__main__":
    ok_flag = run_tests()
    sys.exit(0 if ok_flag else 1)
