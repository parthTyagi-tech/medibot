# 🩺 MediBot — Your AI Health Companion

**MediBot (MediAssist)** is a full-stack, multilingual medical chatbot that combines Retrieval-Augmented Generation (RAG) with real-time voice conversation. It gives users direct, doctor-friend-style answers grounded in a curated medical knowledge base, remembers context across sessions, and lets users talk to it out loud in seven languages.

🔗 **Live demo:** [medibot-22m0.onrender.com](https://medibot-22m0.onrender.com)

---

## ✨ Features

### 🧠 Intelligent, Grounded Answers
- **RAG pipeline** built on **Pinecone** (vector search) + **LangChain**, so medical answers are grounded in an indexed knowledge base rather than hallucinated.
- **Intent classification** routes every message to the right handling path — `medical_query`, `greeting`, `memory_recall`, `general_chat`, or `account_action` — using a lightweight Groq model for fast, cheap classification before the main LLM call.
- **Streaming responses** for a natural, real-time typing experience in both text and voice chat.

### 🗣️ Multilingual Voice Chat
- Real-time voice conversations powered by **LiveKit Agents**, **Deepgram** (STT + TTS), and **Groq's Llama 3.3 70B**.
- Supports **7 languages** out of the box: English, Spanish, French, German, Italian, Dutch, and Japanese — each with its own STT model, TTS voice, and localized system prompt/greeting.
- Voice Activity Detection (Silero VAD) with configurable turn-taking/endpointing for natural conversation flow.
- Live transcript streaming to the browser alongside spoken audio.
- Smart agent dispatch logic that reuses active LiveKit sessions and cleans up stale/zombie dispatches automatically.

### 👤 Personalized & Persistent
- **Google OAuth** and traditional email/password authentication (with secure OTP-based password reset via email).
- **Per-user memory** that updates automatically in the background as conversations happen, so MediBot recalls relevant details about the user over time.
- **Chat session history** with automatic, AI-generated conversation titles and summaries — old sessions are condensed so context is never lost, even after hundreds of messages.
- All memory/title updates run in background threads so they never block the response the user is waiting on.

### ⚙️ Production-Minded Architecture
- Clean separation between the web app (`app.py`), voice worker (`voice_worker.py` / `voice_agent.py`), and RAG/data layer (`research/`, `store_index.py`).
- SQLAlchemy models for users, chat sessions, and messages.
- Deployed on **Render** with Gunicorn.

---

## 🏗️ Architecture Overview

```
┌─────────────┐      HTTP/OAuth       ┌────────────────────┐
│   Browser   │◄─────────────────────►│   Flask App (app.py)│
└─────┬───────┘                       └─────────┬──────────┘
      │ LiveKit WebRTC                           │
      ▼                                          ▼
┌─────────────┐    /voice_chat (HTTP)    ┌───────────────────┐
│ Voice Worker │◄────────────────────────►│  RAG Chain         │
│ (LiveKit     │                          │  LangChain +       │
│  Agent)      │                          │  Pinecone +        │
│  Deepgram    │                          │  Groq (Llama 3.3)  │
│  STT/TTS     │                          └─────────┬──────────┘
└─────────────┘                                     │
                                                      ▼
                                            ┌───────────────────┐
                                            │  SQLite / SQLAlchemy│
                                            │  Users, Sessions,   │
                                            │  Messages, Memory   │
                                            └───────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask, Flask-Login, Flask-Dance (Google OAuth), Flask-Mail |
| **LLM / RAG** | LangChain, Groq (Llama 3.3 70B Versatile, Llama 3.1 8B Instant), Pinecone |
| **Voice** | LiveKit Agents, Deepgram (STT + Aura-2 TTS), Silero VAD |
| **Database** | SQLAlchemy (SQLite) |
| **Deployment** | Render, Gunicorn |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Accounts/API keys for: [Pinecone](https://www.pinecone.io/), [Groq](https://groq.com/), [Deepgram](https://deepgram.com/), [LiveKit](https://livekit.io/), and a Google Cloud OAuth client.

### 1. Clone the repo
```bash
git clone https://github.com/parthTyagi-tech/medibot.git
cd medibot
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```env
# Core
SECRET_KEY=your-flask-secret-key
SERVER_NAME=              # optional, leave blank on Render

# Mail (for OTP password reset)
MAIL_PASSWORD=your-gmail-app-password

# LLM / RAG
PINECONE_API_KEY=your-pinecone-key
GROQ_API_KEY=your-groq-key

# Voice (LiveKit + Deepgram)
LIVEKIT_URL=wss://your-livekit-instance
LIVEKIT_API_KEY=your-livekit-key
LIVEKIT_API_SECRET=your-livekit-secret
DEEPGRAM_API_KEY=your-deepgram-key

# Google OAuth
GOOGLE_OAUTH_CLIENT_ID=your-client-id
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
```

> ⚠️ **Security note:** Never commit real API keys or app passwords to version control. Use `.env` (already in `.gitignore`) or your hosting provider's environment variable settings.

### 4. Build the knowledge base
Add your source medical PDFs to the `data/` folder, then run:
```bash
python store_index.py
```
This chunks, embeds, and upserts your documents into the `medical-chatbot` Pinecone index.

### 5. Run locally
```bash
python app.py
```
This starts the Flask app on `http://localhost:5050` **and** automatically launches the LiveKit voice worker (`voice_worker.py`) in the background for local development.

---

## 📁 Project Structure

```
medibot/
├── app.py                # Main Flask application & routes
├── voice_agent.py         # LiveKit agent: STT/LLM/TTS pipeline, multilingual configs
├── voice_worker.py        # Entry point for the LiveKit agent worker
├── livekit_token.py       # LiveKit JWT + room name generation
├── deepgram_tts.py        # Text-to-speech helper for the /tts route
├── store_index.py         # Builds the Pinecone knowledge base from PDFs
├── research/
│   └── src/
│       ├── auth.py         # User, ChatSession, Message models + Google OAuth blueprint
│       ├── memory.py       # Per-user memory read/update logic
│       ├── helper.py       # PDF loading, chunking, embeddings
│       └── intent_classifier.py
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS/JS/assets
├── tests/                  # Test suite
├── data/                   # Source PDFs for the knowledge base
├── requirements.txt
├── Procfile                # Render/Heroku process definition
├── gunicorn.conf.py
└── runtime.txt
```

---

## 🌐 Deployment

MediBot is deployed on **[Render](https://render.com)** using Gunicorn as the WSGI server (see `Procfile` and `gunicorn.conf.py`). To deploy your own instance:

1. Push this repo to your own GitHub account.
2. Create a new **Web Service** on Render, connect the repo.
3. Add all environment variables listed above in the Render dashboard.
4. Set the build command to `pip install -r requirements.txt` and the start command from `Procfile`.
5. For voice chat in production, deploy the LiveKit agent worker (`voice_worker.py`) as a separate background worker/service so it stays connected independently of web request/response cycles.

---

## 🗺️ Roadmap Ideas

- [ ] Add more languages to the voice pipeline
- [ ] Support document upload for user-specific knowledge bases
- [ ] Add unit test coverage for the RAG and intent-classification paths
- [ ] Rate limiting / abuse protection on public routes
- [ ] Migrate from SQLite to PostgreSQL for production scale

---

## 📄 License

See [LICENSE](./LICENSE) for details.

---

## 🙋 Author

Built by [Parth Tyagi](https://github.com/parthTyagi-tech).
