"""
bangla_rag_chatbot.py  —  Bangla RAG Chatbot (interactive CLI)

Bugs fixed vs original:
  - Removed duplicate 'from rag_engine import' inside view_topic_faqs()
  - result['difficulty'] KeyError: now uses result.get('difficulty','N/A')
  - FAQ file path resolved by rag_engine._resolve_faq_path() - works from any cwd
  - sys.stdout/stdin.reconfigure wrapped in try/except for non-TTY envs
"""

import os
import sys
from typing import Optional

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # Python < 3.7 or non-TTY

# Ensure src/ is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import BanglaRAGEngine


class BanglaChatbot:
    def __init__(self, faq_file_path: str):
        """
        Initialize the Bangla RAG Chatbot.

        Args:
            faq_file_path: Path (or hint) to faq_data.json.
                           The engine will search common locations automatically.
        """
        self.rag_engine         = BanglaRAGEngine(faq_file_path)
        self.current_topic_id   = None
        self.conversation_history = []

    # ── Display helpers ───────────────────────────────────────────────────────

    def print_banner(self):
        print("\n" + "=" * 60)
        print("      Bangla RAG Chatbot - Welcome!")
        print("      বাংলা RAG চ্যাটবট - স্বাগতম!")
        print("=" * 60 + "\n")

    def print_main_menu(self):
        print("\n" + "-" * 60)
        print("Main Menu / প্রধান মেনু")
        print("-" * 60)
        print("1. Select a Topic      / বিষয় নির্বাচন করুন")
        print("2. Ask a Question      / প্রশ্ন জিজ্ঞাসা করুন")
        print("3. View All Topics     / সমস্ত বিষয় দেখুন")
        print("4. View Topic FAQs     / বিষয়ের সব FAQ দেখুন")
        print("5. Conversation History/ চ্যাট ইতিহাস দেখুন")
        print("6. Exit                / প্রোগ্রাম বন্ধ করুন")
        print("-" * 60)

    # ── Topic selection ───────────────────────────────────────────────────────

    def display_topics(self):
        """Print all available topics and return them."""
        topics = self.rag_engine.get_all_topics()
        print("\n" + "=" * 60)
        print("Available Topics / উপলব্ধ বিষয়")
        print("=" * 60)
        for topic_id, name, name_en in topics:
            count = len(self.rag_engine.get_faqs_by_topic(topic_id))
            marker = " <-- selected" if topic_id == self.current_topic_id else ""
            print(f"  {topic_id}. {name} ({name_en})  [{count} FAQs]{marker}")
        print("=" * 60)
        return topics

    def select_topic(self) -> Optional[int]:
        """Prompt user to choose a topic; return the topic_id or None."""
        topics = self.display_topics()
        valid_ids = {t[0] for t in topics}

        while True:
            try:
                raw = input("\nEnter topic number (or 0 to cancel): ").strip()
                num = int(raw)
                if num == 0:
                    return None
                if num in valid_ids:
                    self.current_topic_id = num
                    chosen = next(t for t in topics if t[0] == num)
                    print(f"\nSelected: {chosen[1]} ({chosen[2]})")
                    return num
                print(f"  Please enter a number between 1 and {max(valid_ids)}.")
            except ValueError:
                print("  Please enter a valid number.")

    # ── Q&A ───────────────────────────────────────────────────────────────────

    def ask_question(self):
        """Prompt for a question, search, and display the answer."""
        topic_hint = ""
        if self.current_topic_id:
            topics     = self.rag_engine.get_all_topics()
            chosen     = next((t for t in topics if t[0] == self.current_topic_id), None)
            topic_hint = f" [{chosen[1]}]" if chosen else ""

        print(f"\n{'─'*60}")
        print(f"Ask your question{topic_hint} (leave blank to cancel):")
        question = input(">>> ").strip()

        if not question:
            print("  Cancelled.")
            return

        result = self.rag_engine.search(question, self.current_topic_id)

        print("\n" + "=" * 60)
        if result["found"]:
            print("Answer found! / উত্তর পাওয়া গেছে!")
            print("=" * 60)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nTopic      : {result['topic']}")
            print(f"Difficulty : {result.get('difficulty', 'N/A')}")
            print(f"Similarity : {result['similarity']:.1%}")
            print(f"Matched Q  : {result.get('matched_question', '')}")
        else:
            print("No direct answer found / সরাসরি উত্তর পাওয়া যায়নি")
            print("=" * 60)
            print(f"\n{result['answer']}")
            if result["similarity"] > 0:
                print(f"\nBest similarity was only {result['similarity']:.1%} — "
                      "try selecting a specific topic first (option 1).")
        print("=" * 60)

        # Record in history
        self.conversation_history.append({
            "question": question,
            "answer":   result["answer"],
            "found":    result["found"],
            "topic":    result.get("topic") or "—",
        })

    # ── Browse FAQs ───────────────────────────────────────────────────────────

    def view_topic_faqs(self):
        """Display every Q&A for the currently selected topic."""
        if not self.current_topic_id:
            print("\n  Please select a topic first (option 1).")
            return

        faqs   = self.rag_engine.get_faqs_by_topic(self.current_topic_id)
        topics = self.rag_engine.get_all_topics()
        chosen = next((t for t in topics if t[0] == self.current_topic_id), None)

        if not faqs:
            print(f"\n  No FAQs found for topic id={self.current_topic_id}.")
            return

        print("\n" + "=" * 60)
        print(f"All FAQs: {chosen[1]} ({chosen[2]})" if chosen else "All FAQs")
        print("=" * 60)
        for i, faq in enumerate(faqs, 1):
            print(f"\n{i}. Q: {faq['question']}")
            print(f"   A: {faq['answer']}")
            print(f"   Difficulty: {faq.get('difficulty', 'N/A')}")
        print()

    # ── History ───────────────────────────────────────────────────────────────

    def view_conversation_history(self):
        """Print all previous Q&A pairs."""
        if not self.conversation_history:
            print("\n  No conversation history yet.")
            return

        print("\n" + "=" * 60)
        print("Conversation History / কথোপকথন ইতিহাস")
        print("=" * 60)
        for i, entry in enumerate(self.conversation_history, 1):
            status = "Found" if entry["found"] else "Not found"
            print(f"\n{i}. [{status}] Topic: {entry['topic']}")
            print(f"   Q: {entry['question']}")
            # Safe truncation - avoid IndexError on short answers
            ans_preview = entry["answer"][:120]
            if len(entry["answer"]) > 120:
                ans_preview += "..."
            print(f"   A: {ans_preview}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.print_banner()
        print("This chatbot answers questions in Bangla across 5 topics:")
        print("Education | Health | Travel | Technology | Sports\n")

        DISPATCH = {
            "1": self.select_topic,
            "2": self.ask_question,
            "3": self.display_topics,
            "4": self.view_topic_faqs,
            "5": self.view_conversation_history,
        }

        while True:
            self.print_main_menu()
            choice = input("Your choice (1-6): ").strip()

            if choice == "6":
                print("\nThank you! See you again. / ধন্যবাদ! আবার দেখা হবে।\n")
                break
            elif choice in DISPATCH:
                DISPATCH[choice]()
                # After selecting a topic (1), immediately offer to ask a question
                if choice == "1" and self.current_topic_id is not None:
                    ask_now = input("\nAsk a question in this topic now? (y/n): ").strip().lower()
                    if ask_now == "y":
                        self.ask_question()
            else:
                print("  Invalid choice. Please enter 1-6.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Pass a hint; _resolve_faq_path() in rag_engine will find the real file
    faq_hint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "faq_data.json"
    )
    chatbot = BanglaChatbot(faq_hint)
    chatbot.run()


if __name__ == "__main__":
    main()
