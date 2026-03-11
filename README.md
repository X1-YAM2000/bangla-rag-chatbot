# Bangla RAG Chatbot
### বাংলা RAG চ্যাটবট

A fully offline Bangla FAQ chatbot using TF-IDF retrieval with optional LLM answer refinement and a FastAPI REST backend.

---

## Features

- 5 Bangla topics: Education, Health, Travel, Technology, Sports
- TF-IDF + cosine similarity retrieval (no internet, no GPU needed)
- Optional LLM refinement via Ollama (free/local) or OpenAI
- FastAPI REST API with Swagger docs
- Topic filtering and difficulty metadata
- Fallback response when no match found
- Full test suite (7 tests, all pass)

---

## Project Structure

```
M17/
├── src/
│   ├── rag_engine.py           # Core retrieval + LLM refinement engine
│   ├── bangla_rag_chatbot.py   # Interactive CLI chatbot
│   ├── api.py                  # FastAPI REST backend
│   ├── demo_chatbot.py         # Non-interactive demo script
│   └── test_chatbot.py         # Automated test suite (7 tests)
├── data/
│   └── faq_data.json           # 5 topics x 3 FAQs = 15 Q&A pairs
├── requirements.txt
├── quick_start.bat             # Windows launcher
├── quick_start.sh              # Linux/Mac launcher
├── run_chatbot.bat             # Direct Windows runner
└── run_chatbot.ps1             # Direct PowerShell runner
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.7+, scikit-learn, numpy, fastapi, uvicorn

---

## Usage

### Option A: Quick Start (recommended)

**Windows:**
```bat
quick_start.bat
```

**Linux / Mac / WSL:**
```bash
bash quick_start.sh
```

Both scripts install dependencies then show a menu:
```
1. Interactive Chatbot
2. Run Tests
3. Run Demo
4. Start FastAPI Server (port 8000)
```

### Option B: Run directly

```bash
# Interactive chatbot
python src/bangla_rag_chatbot.py

# Tests
python src/test_chatbot.py

# Demo (non-interactive)
python src/demo_chatbot.py

# FastAPI server
uvicorn src.api:app --reload --port 8000
```

---

## FastAPI Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check |
| GET | `/topics` | List all 5 topics |
| GET | `/topics/{id}/faqs` | FAQs for a topic |
| POST | `/chat` | Ask a question (JSON) |
| GET | `/search?q=...` | Quick GET search |

**Interactive docs:** http://localhost:8000/docs

**Example POST /chat:**
```json
{
  "question": "প্রতিদিন কত পানি পানো উচিত?",
  "topic_id": 2,
  "threshold": 0.1
}
```

**Response:**
```json
{
  "found": true,
  "answer": "সাধারণত একজন প্রাপ্তবয়স্ক...",
  "raw_answer": "সাধারণত একজন প্রাপ্তবয়স্ক...",
  "similarity": 0.925,
  "topic": "স্বাস্থ্য",
  "difficulty": "সহজ",
  "matched_question": "প্রতিদিন কতটা পানি পানো উচিত?"
}
```

---

## LLM Answer Refinement (Optional)

The engine can rewrite retrieved FAQ answers in more fluent Bangla using a local or cloud LLM.

### Ollama (free, offline)

```bash
# Install: https://ollama.com/download
ollama pull llama3

# Set env vars, then run
export BANGLA_LLM_PROVIDER=ollama
export BANGLA_LLM_MODEL=llama3
pip install openai
python src/bangla_rag_chatbot.py
```

### OpenAI

```bash
export BANGLA_LLM_PROVIDER=openai
export BANGLA_LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
pip install openai
python src/bangla_rag_chatbot.py
```

When active, the API response includes both `answer` (refined) and `raw_answer` (original FAQ text).

---

## Test Results

```
Test 1: Initialization        PASS  (index built, 15 FAQs, 5 topics)
Test 2: Topic Loading         PASS  (all 5 topics, >=3 FAQs each)
Test 3: Search No Filter      PASS  (5/5 queries answered, avg 72%)
Test 4: Topic Filtering       PASS  (filter restricts results to topic)
Test 5: Threshold & Fallback  PASS  (fallback response on no match)
Test 6: get_faqs_by_topic     PASS  (all topics, invalid ID = [])
Test 7: Result Dict Keys      PASS  (all keys present, found + not-found)

ALL 7 TESTS PASSED
```

---

## How It Works

```
User question
     |
     v
TF-IDF vectorizer (char n-grams, trained on FAQ questions)
     |
     v
Cosine similarity against all FAQ vectors
     |
     +-- topic filter applied (if topic_id provided)
     |
     v
Best match >= threshold?
     |
     +-- YES: return FAQ answer  --> (optional) LLM refines to fluent Bangla
     |
     +-- NO:  return fallback message
```

---

## FAQ Data Format

```json
{
  "topics": [
    {
      "id": 1,
      "name": "শিক্ষা",
      "name_en": "Education",
      "faqs": [
        {
          "id": 1,
          "question": "একটি ভালো শিক্ষা প্রতিষ্ঠান কীভাবে নির্বাচন করা উচিত?",
          "answer": "...",
          "topic": "শিক্ষা",
          "difficulty": "মাঝারি"
        }
      ]
    }
  ]
}
```

---

