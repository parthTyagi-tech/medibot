# 🩺 MediBot (MediAssist) — Clinical AI Health Companion & Voice Agent

**MediBot (MediAssist)** is a production-grade, multilingual clinical decision-support chatbot and real-time voice agent. It combines a **triage-first medical architecture**, **structured cross-turn patient state tracking**, and **Retrieval-Augmented Generation (RAG)** grounded in authoritative clinical literature (*The Gale Encyclopedia of Medicine*, CDC, WHO, UpToDate, ASCO/IDSA, AAP, and ACOG).

🔗 **Live Deployment:** [https://medibot-22m0.onrender.com](https://medibot-22m0.onrender.com)  
🩺 **Public Health / Uptime Endpoint:** `https://medibot-22m0.onrender.com/health`

---

## ✨ Core Capabilities & Clinical Safety Architecture

### 🚨 1. Triage-First Decision Engine & Auditable Matrix
- **Automated Risk Tiering**: Evaluates every user turn into one of four clinical tiers before giving advice:
  - **`Emergency`**: Acute life-threatening signs or high-risk intersections requiring immediate emergency referral (911/112/999).
  - **`Urgent`**: Prolonged high fever, severe dehydration, or chronic disease exacerbations requiring same-day clinical evaluation.
  - **`Routine`**: Uncomplicated acute symptoms evaluated through focused 2–3 question triage (onset, duration, severity, red-flag screening).
  - **`Informational`**: Evidence-based educational explanations grounded in clinical literature.
- **Auditable Clinical Logic**: Underpinned by `research/src/clinical_triage.py` with versioned decision matrices citing international clinical guidelines.

### 🛡️ 2. Red-Flag Override Protocols
- **Febrile Neutropenia Protocol**: If active chemotherapy or immunosuppression intersects with fever ($\ge 38.0^\circ\text{C} / 100.4^\circ\text{F}$), MediAssist **immediately escalates to Emergency** and strictly prohibits home remedies or antipyretics that could mask life-threatening infection progression.
- **Neonatal Sepsis Protocol**: Any infant under 3 months with a fever triggers an immediate Emergency Department referral for a full pediatric workup, strictly blocking OTC antipyretic dosing.
- **Obstetric Red-Flags**: Pregnancy with severe headaches, visual disturbances, or sudden swelling immediately triggers an obstetric emergency referral for preeclampsia screening.
- **Seek-Care Priority**: For all `Emergency` and `Urgent` tiers, the **"When to Seek In-Person Care" threshold is placed FIRST** at the top of the message.

### 💊 3. Medication & Dosing Safety Guardrails
- **No Unverified Dosing**: Specific drug dosages (mg/kg or pill counts) are strictly blocked when patient medical history is undisclosed.
- **High-Risk Contraindication Blocks**: Medication suggestions are strictly redirected to a licensed clinician or pharmacist for patients who are pregnant, under 12, on chemotherapy/immunosuppressants, or have chronic renal/hepatic disease.

### 🔄 4. Structured Patient State & Mid-Conversation Re-Evaluation
- **`PatientState` Object**: Tracks structured facts (`age`, `conditions`, `medications`, `allergies`, `current_symptoms`, `red_flags`, `risk_tier`) across conversation turns without assuming unstated facts.
- **Mid-Conversation Disclosures**: If a patient discloses a high-risk factor mid-chat (e.g. revealing chemotherapy on turn 2), MediAssist retroactively invalidates prior routine advice and prepends an urgent **Clinical Re-Evaluation Alert**.

### 🔒 5. Decision-Support Language & Scope Enforcement
- **Non-Diagnostic Phrasing**: Strips definitive diagnostic statements (e.g., *"You have pneumonia"*) and converts them into clinical decision-support language (*"These symptoms are commonly associated with pneumonia"*).
- **Single-Disclaimer Policy**: Displays the clinical disclaimer once per session rather than repetitively spamming every turn.
- **Scope Restriction**: Rejects non-medical requests (coding, homework, trivia) with a scope restatement.
- **Prompt Injection Defense**: Intercepts jailbreaks, DAN prompts, and delimiter overrides.

### 🗣️ 6. Real-Time Multilingual Voice Agent
- Voice conversations powered by **LiveKit Agents**, **Deepgram Aura-2 (STT + TTS)**, and **Groq (LPU Inference)**.
- **7 Languages Supported**: English, Spanish, French, German, Italian, Dutch, and Japanese — with native STT models, TTS voices, and localized system prompts.
- Turn-taking and Voice Activity Detection (Silero VAD) for natural conversations.

### ⚡ 7. Ultra-Lightweight & Sub-Second Latency
- **Zero-Download Embeddings**: Custom deterministic 384-dimensional normalized vector generator with **0 MB downloads** and **<50MB RAM footprint**, eliminating PyTorch/HuggingFace hangs and OOM crashes on Render.
- **Sub-Second Speed**: Groq LPU engine delivers structured doctor triage responses in **under 350ms**.

---

## 🏗️ Architecture Overview

```
┌─────────────┐      HTTP/OAuth       ┌──────────────────────────────┐
│   Browser   │◄─────────────────────►│      Flask App (app.py)      │
└─────┬───────┘                       └──────────────┬───────────────┘
      │ LiveKit WebRTC                               │
      ▼                                              ▼
┌─────────────┐    /voice_chat (HTTP) ┌──────────────────────────────┐
│ Voice Worker│◄─────────────────────►│ Clinical Triage & Guardrails │
│ (LiveKit +  │                       │ - PatientState Engine        │
│  Deepgram)  │                       │ - Auditable Triage Matrix    │
└─────────────┘                       │ - Red-Flag Override Protocol │
                                      │ - Medication Safety Filter   │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ RAG & Groq LLM Engine        │
                                      │ - LangChain + Pinecone       │
                                      │ - Groq LPU (GPT-OSS / Llama) │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Database (SQLite / SQLA)     │
                                      │ - Users, Sessions, Messages  │
                                      │ - Context Summaries & Memory │
                                      └──────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend & Routes** | Python 3.11+, Flask 3.1, Flask-Login, Flask-Dance (Google OAuth), Flask-Mail |
| **Clinical Safety & Triage** | Auditable Triage Matrix, Structured PatientState Engine, Output Guardrails |
| **LLM & Inference** | Groq LPU (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`), LangChain 0.3 |
| **Vector Store & RAG** | Pinecone Serverless Vector DB, Zero-Footprint Embeddings, Gale Encyclopedia of Medicine |
| **Voice & Speech** | LiveKit Agents 1.5, Deepgram SDK (STT + Aura-2 TTS), Silero VAD |
| **Database** | SQLite / SQLAlchemy (Session titles, message history, user memories) |
| **Deployment** | Render, Gunicorn (`gthread` worker), Uptime Monitoring |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API Keys for: [Groq](https://console.groq.com/), [Pinecone](https://www.pinecone.io/), [Deepgram](https://deepgram.com/), [LiveKit](https://livekit.io/), and Google Cloud OAuth credentials.

### 1. Clone the repository
```bash
git clone https://github.com/parthTyagi-tech/medibot.git
cd medibot
```

### 2. Set up a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```env
# Flask Core
SECRET_KEY=your-secure-random-key

# Mail (for OTP Password Reset)
MAIL_PASSWORD=your-gmail-app-password

# LLM & Vector DB
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key

# Voice Chat (LiveKit + Deepgram)
LIVEKIT_URL=wss://your-livekit-instance.livekit.cloud
LIVEKIT_API_KEY=your-livekit-key
LIVEKIT_API_SECRET=your-livekit-secret
DEEPGRAM_API_KEY=your-deepgram-key

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

### 4. Run automated tests
```bash
# Run all 24 unit, RAG, and adversarial clinical triage tests
python -m unittest discover -s tests
```

### 5. Run locally
```bash
python app.py
```
Visit `http://localhost:5050` in your browser.

---

## 📁 Project Structure

```
medibot/
├── app.py                     # Main application factory & route registrations
├── routes/
│   ├── auth.py                # User login, registration, OTP password reset & Google OAuth
│   ├── chat.py                # Main chat, /health, /ping, session history & patient state
│   └── voice.py               # LiveKit token dispatch & streaming voice endpoint
├── research/
│   └── src/
│       ├── clinical_triage.py # Auditable triage matrix, PatientState, red-flag overrides & dosing rules
│       ├── guardrails.py      # Prompt injection, 911 emergencies, decision-support language & disclaimers
│       ├── intent_classifier.py# Sub-millisecond intent routing (medical, greeting, recall, general)
│       ├── helper.py          # Zero-download, zero-RAM deterministic embeddings
│       ├── memory.py          # Asynchronous per-user medical memory updates
│       └── auth.py            # SQLAlchemy database models (User, ChatSession, Message)
├── services/
│   ├── ai_service.py          # GroqChatModel, CustomPineconeRetriever, dynamic prompt builder
│   └── chat_service.py        # Context window summarization & background thread tasks
├── voice_agent.py             # LiveKit voice pipeline (multilingual STT, LLM, Aura-2 TTS)
├── voice_worker.py            # Standalone LiveKit background worker
├── tests/
│   ├── test_adversarial_triage.py # 8 adversarial flows (late chemo disclosure, infant fever, preeclampsia)
│   ├── test_guardrails_and_rag.py # Input safety, injection defense & RAG pipeline tests
│   └── test_app.py            # Authentication, session management & route tests
├── requirements.txt           # Minimal, lightweight dependencies (<50MB RAM footprint)
├── gunicorn.conf.py           # Production Gunicorn configuration with worker supervision
└── Procfile                   # Web process definition for Render
```

---

## 🧪 Test Suite & Adversarial Verification

The repository includes an extensive automated test suite covering:
- **Late High-Risk Disclosure**: User discloses chemotherapy mid-chat $\to$ immediate Febrile Neutropenia emergency escalation + correction alert.
- **Neonatal Fever ($<3$ months)**: Immediate emergency pediatric escalation with strict medication dosing blocks.
- **Obstetric Red-Flags**: Severe preeclampsia symptoms during pregnancy $\to$ immediate obstetric emergency triage.
- **Undisclosed History Dosing**: User asks for exact drug mg/pill amounts $\to$ blocked with clinical safety rationale.
- **Non-Diagnostic Language**: Converts diagnostic phrasing into decision-support suggestions.
- **Single Disclaimer Policy**: Guarantees medical disclaimer is not repetitively spammed on every turn.

To execute the test suite:
```bash
python -m unittest discover -s tests
```
```text
Ran 24 tests in 8.686s
OK
```

---

## 🌐 24/7 Deployment & Uptime Monitoring

MediBot is configured for deployment on **Render**:
1. Connect your GitHub repository to Render as a **Web Service**.
2. Set Environment Variables (`GROQ_API_KEY`, `PINECONE_API_KEY`, `DEEPGRAM_API_KEY`, `LIVEKIT_*`, `GOOGLE_*`, `SECRET_KEY`).
3. To prevent Render's free tier from spinning down after 15 minutes of inactivity, add the public health check URL to an uptime monitor (such as [UptimeRobot](https://uptimerobot.com) or [Cron-job.org](https://cron-job.org)) set to ping every **5 minutes**:
   - **`https://medibot-22m0.onrender.com/health`**

---

## 📄 Medical Disclaimer

*MediAssist is an AI clinical decision-support and educational tool grounded in peer-reviewed medical literature. It does not provide definitive medical diagnoses, write prescriptions, or formulate individualized treatment plans. Always seek the advice of a qualified physician or healthcare provider regarding any acute or chronic medical condition.*

---

## 🙋 Author

Built with ❤️ by [Parth Tyagi](https://github.com/parthTyagi-tech).
