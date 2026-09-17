# 🩺 MediBot (MediAssist) — Clinical AI Health Companion & Voice Agent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Groq LPU](https://img.shields.io/badge/Groq-LPU%20Inference-F05A28?style=flat)](https://groq.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless%20Vector%20DB-000000?style=flat&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![LiveKit](https://img.shields.io/badge/LiveKit-WebRTC%20Voice%20Agents-000000?style=flat&logo=livekit&logoColor=white)](https://livekit.io/)
[![Deepgram](https://img.shields.io/badge/Deepgram-Aura--2%20STT%20%2F%20TTS-13EF93?style=flat&logoColor=black)](https://deepgram.com/)
[![Redis Streams](https://img.shields.io/badge/Redis-Streams%20%26%20DLQ-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-30%2F30%20Passing-brightgreen?style=flat&logo=githubactions&logoColor=white)](tests/)
[![Render](https://img.shields.io/badge/Render-Live%20Deployment-46E3B7?style=flat&logo=render&logoColor=white)](https://medibot-22m0.onrender.com)

**MediBot (MediAssist)** is an enterprise-grade, multilingual clinical decision-support chatbot and real-time voice agent. It unites a **deterministic triage-first medical architecture**, **structured cross-turn patient state tracking**, and **Retrieval-Augmented Generation (RAG)** grounded in authoritative medical literature (*The Gale Encyclopedia of Medicine*, CDC, WHO, UpToDate, ASCO/IDSA, AAP, and ACOG). 

Engineered for production resilience, it incorporates **decoupled asynchronous reliability via Redis Streams with Dead-Letter Queues (DLQ)**, **idempotency deduplication**, **multi-provider transactional email (Brevo HTTP API + SMTP)**, and a **zero-download deterministic embedding pipeline** running stably within **<50MB RAM**.

🔗 **Live Deployment:** [https://medibot-22m0.onrender.com](https://medibot-22m0.onrender.com)  
🩺 **Public Health / Uptime Endpoint:** [https://medibot-22m0.onrender.com/health](https://medibot-22m0.onrender.com/health)

---

## ✨ Core Capabilities & Clinical Safety Architecture

### 🚨 1. Triage-First Decision Engine & Auditable Matrix
- **Automated Risk Tiering**: Every patient turn is evaluated and mapped into one of four clinical tiers before formulating advice:
  - **`Emergency`**: Acute life-threatening red flags or critical intersections requiring immediate emergency referral (911/112/999).
  - **`Urgent`**: Prolonged high fever, severe dehydration, or chronic disease exacerbations requiring same-day clinical evaluation.
  - **`Routine`**: Uncomplicated acute symptoms evaluated through focused 2–3 question triage (onset, duration, severity, red-flag screening).
  - **`Informational`**: Evidence-based educational explanations grounded in clinical literature.
- **Auditable Clinical Logic**: Underpinned by `research/src/clinical_triage.py` with versioned decision matrices citing international clinical practice guidelines.

### 🛡️ 2. Red-Flag Override Protocols
- **Febrile Neutropenia Protocol**: If active chemotherapy or immunosuppression intersects with fever ($\ge 38.0^\circ\text{C} / 100.4^\circ\text{F}$), MediAssist **immediately escalates to Emergency** and strictly prohibits home remedies or antipyretics that could mask life-threatening infection progression.
- **Neonatal Sepsis Protocol**: Any infant under 3 months presenting with fever triggers an immediate Emergency Department referral for a full pediatric workup, strictly blocking OTC antipyretic dosing.
- **Obstetric Red-Flags**: Pregnancy presenting with severe headaches, visual disturbances, or sudden swelling immediately triggers an obstetric emergency referral for preeclampsia screening.
- **Seek-Care Priority**: For all `Emergency` and `Urgent` tiers, the **"When to Seek Immediate In-Person Care" threshold is placed FIRST** at the very top of the response.

### 💊 3. Medication & Dosing Safety Guardrails
- **No Unverified Dosing**: Specific drug dosages (mg/kg or pill counts) are strictly blocked when patient medical history or weight is undisclosed.
- **Pediatric Safety Blocks**: Medication dosing requests for patients under 12 are strictly blocked; users are directed to pediatricians for weight-based clinical evaluation.
- **High-Risk Contraindication Blocks**: Medication suggestions are strictly redirected to licensed clinicians or pharmacists for patients who are pregnant, under 12, on chemotherapy/immunosuppressants, or have chronic renal/hepatic impairment.

### 🔄 4. Structured Patient State & Mid-Conversation Re-Evaluation
- **`PatientState` Object**: Tracks structured clinical facts (`age`, `conditions`, `medications`, `allergies`, `current_symptoms`, `red_flags`, `risk_tier`) across conversation turns without assuming unstated facts or suffering context drift.
- **Mid-Conversation Late Disclosures**: If a patient discloses a high-risk factor mid-chat (e.g., revealing active chemotherapy on turn 2), MediAssist retroactively invalidates prior routine advice and prepends an urgent **Clinical Re-Evaluation Alert**.

### ⚡ 5. Decoupled Asynchronous Reliability (Redis Streams & DLQ)
- **Zero Request-Thread Blocking**: Long-running background operations (per-user medical memory extraction, automated conversation titling, and 8-turn conversation summarization) are offloaded asynchronously.
- **Redis Streams Consumer Groups**: Tasks are published to Redis Streams (`medical-background-tasks`) and consumed by isolated worker processes (`task_worker.py`).
- **Phase C Idempotency Deduplication**: Deterministic SHA-256 idempotency keys (`idemp:{type}:{id}:{hash}`) prevent duplicate job processing and eliminate redundant LLM token spend.
- **Exponential Backoff & Dead-Letter Queue (DLQ)**: Retries transient failures up to 3 times with exponential backoff. Permanently failed jobs are routed to `medical-tasks-dlq` for inspection without dropping customer requests.
- **Graceful Offline Fallback**: If Redis is offline or unconfigured, the dispatcher automatically executes jobs using a local thread pool (`ThreadPoolExecutor`), ensuring zero crashes in standalone or development environments.

### 🔒 6. Medico-Legal Language & Scope Enforcement
- **Non-Diagnostic Phrasing**: Deterministically rewrites definitive diagnostic declarations (*"You have pneumonia"*) into compliant decision-support suggestions (*"These symptoms are commonly associated with pneumonia"*).
- **Single-Disclaimer Policy**: Displays the medical disclaimer once per session rather than repetitively cluttering every conversation turn.
- **Scope Restriction**: Instantly rejects non-medical queries (coding, homework, trivia) with a courteous scope clarification.
- **Prompt Injection Defense**: Intercepts jailbreaks, DAN templates, and delimiter manipulation attempts in 0ms using pre-compiled regex automatons.

### 🗣️ 7. Real-Time Multilingual Voice Agent
- Ultra-low latency voice conversations powered by **LiveKit Agents 1.5**, **Deepgram Aura-2 (STT + TTS)**, and **Groq (LPU Inference)**.
- **7 Languages Supported**: English, Spanish, French, German, Italian, Dutch, and Japanese — with native STT models, localized TTS voices, and translated system prompts.
- Turn-taking and Voice Activity Detection (Silero VAD) for natural, interruptible conversations.

### 📬 8. Resilient Multi-Provider Transactional Email
- Supports **Brevo (Sendinblue) REST HTTP API** for one-time password (OTP) resets, bypassing cloud host SMTP port restrictions (e.g., Render outbound port 25/587 blocks).
- Gracefully falls back to **Resend**, **SendGrid**, or standard **Flask-Mail SMTP**.

### ⚡ 9. Ultra-Lightweight & Sub-Second Latency
- **Zero-Download Embeddings**: Custom deterministic 384-dimensional normalized vector generator with **0 MB downloads** and **<50MB RAM footprint**, eliminating PyTorch/HuggingFace hangs and OOM crashes on Render.
- **Sub-Second Speed**: Groq LPU engine delivers structured doctor triage responses in **under 350ms** with automatic failover between primary (`openai/gpt-oss-20b`) and secondary (`openai/gpt-oss-120b`) models.

---

## 🏗️ Architecture Overview

```
                                 ┌─────────────────────────────────────────┐
                                 │          Client Browser / App           │
                                 └──────────────┬──────────────────┬───────┘
                                                │                  │
                           HTTPS (Chat / Auth)  │                  │  WebRTC (Voice Audio)
                                                ▼                  ▼
                    ┌───────────────────────────────┐  ┌───────────────────────────────┐
                    │      Flask Application        │  │ LiveKit Voice Worker (Agent)  │
                    │   (app.py + routes/chat.py)   │  │  - Silero VAD                 │
                    └──────────────┬────────────────┘  │  - Deepgram STT + Aura-2 TTS  │
                                   │                   │  - Groq LPU Ultra-Low Latency │
                                   │                   └───────────────┬───────────────┘
                                   ▼                                   │
                    ┌───────────────────────────────┐                  │
                    │ Input Guardrails & Intent     │◄─────────────────┘
                    │ - Regex Injection Tripwire    │   HTTP /voice_chat
                    │ - 0ms Fast Intent Classifier  │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ Clinical Triage & State Engine│
                    │ - Structured PatientState     │
                    │ - Auditable Triage Matrix     │
                    │ - Febrile Neutropenia / Sepsis│
                    │ - Pediatric Dosing Blocks     │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ RAG Retrieval & Groq LLM      │
                    │ - Zero-RAM 384D Embeddings    │
                    │ - Pinecone Serverless DB      │
                    │ - Dual-Model Auto Failover    │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ Output Guardrails & Delivery  │
                    │ - Medico-Legal Sanitization   │
                    │ - Non-Diagnostic Rewriting    │
                    └──────────────┬────────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             │ Synchronous Response                      │ Asynchronous Tasks
             ▼                                           ▼
┌───────────────────────────────┐       ┌─────────────────────────────────┐
│     Client HTTP Response      │       │ Task Dispatcher (task_dispatch) │
│         (< 350ms)             │       │ - SHA-256 Idempotency Check     │
└───────────────────────────────┘       └────────────────┬────────────────┘
                                                         │
                                        ┌────────────────┴────────────────┐
                                        │ Redis Available?                │
                                        ├──► [YES]: Enqueue to Stream     │
                                        │           (medical-background)  │
                                        │                     │           │
                                        │                     ▼           │
                                        │           ┌───────────────────┐ │
                                        │           │ task_worker.py    │ │
                                        │           │ - 3x Backoff Retry│ │
                                        │           │ - DLQ Routing     │ │
                                        │           └─────────┬─────────┘ │
                                        │                     │           │
                                        └──► [NO]:  Execute in ThreadPool │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ SQLite / SQLAlchemy DB    │
                                                │ - User Medical Memory     │
                                                │ - Session Titles & Summary│
                                                └───────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Description |
| :--- | :--- | :--- |
| **Backend & Routing** | Python 3.11+, Flask 3.1.1, Werkzeug ProxyFix | High-throughput web routing, secure session cookies, TLS proxy headers |
| **Authentication & Users** | Flask-Login 0.6.3, Flask-Dance 7.1.0, SQLAlchemy | Session auth, Google OAuth 2.0, multi-session management, patient history |
| **Clinical Safety & Triage** | Custom Clinical Rules Engine, `PatientState` | CDC/WHO/UpToDate auditable triage matrix, pediatric dosing blocks, red-flag overrides |
| **LLM & Inference** | Groq LPU (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`), LangChain 0.3 | Sub-350ms inference, resilient dual-model failover, dynamic clinical prompt construction |
| **Vector Database & RAG** | Pinecone 7.3 Serverless, Custom Normalized Embeddings | Grounded in *The Gale Encyclopedia of Medicine*, zero-download 384D deterministic vectors |
| **Async Reliability & Queue** | Redis Streams, Consumer Groups, Dead-Letter Queue (DLQ) | Offloaded memory updates, session titling, and summarization with retry backoff & DLQ |
| **Voice & Speech** | LiveKit Agents 1.5, Deepgram SDK 7.3, Silero VAD | Real-time WebRTC voice pipeline, Deepgram Aura-2 STT/TTS across 7 languages |
| **Transactional Email** | Brevo (Sendinblue) HTTP API, Resend, Flask-Mail | Multi-provider OTP delivery bypassing cloud SMTP port blocks |
| **Production Server** | Gunicorn (`gthread`), Dual Subprocess Supervisors | Memory-optimized worker configuration supervising Voice and Task worker processes |
| **Deployment & Uptime** | Render, Uptime Monitoring (`/health`, `/ping`) | Cloud container deployment with persistent health and ping endpoints |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- API Keys for: [Groq](https://console.groq.com/), [Pinecone](https://www.pinecone.io/), [Deepgram](https://deepgram.com/), [LiveKit](https://livekit.io/), and Google Cloud OAuth credentials.
- *(Optional)* Redis server for Redis Streams background task queue (falls back automatically to in-memory ThreadPool if absent).
- *(Optional)* Brevo (Sendinblue) API Key for OTP emails over HTTP.

### 1. Clone the repository
```bash
git clone https://github.com/parthTyagi-tech/medibot.git
cd medibot
```

### 2. Set up a virtual environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```env
# Flask Core
SECRET_KEY=your-secure-random-key-here
FLASK_ENV=development
PORT=5050

# LLM & Vector Database
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key

# Voice Chat (LiveKit + Deepgram)
LIVEKIT_URL=wss://your-livekit-instance.livekit.cloud
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
DEEPGRAM_API_KEY=your-deepgram-api-key

# Google OAuth 2.0
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Asynchronous Task Queue (Optional - defaults to in-memory fallback)
REDIS_URL=redis://localhost:6379/0

# Transactional Email (Brevo HTTP API or SMTP)
BREVO_API_KEY=your-brevo-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-gmail-app-password
MAIL_DEFAULT_SENDER=your-email@gmail.com

# Process Management Flags
ENABLE_VOICE_WORKER=true
ENABLE_TASK_WORKER=true
```

### 4. Run automated tests
```bash
# Execute all 30 unit, RAG, adversarial clinical triage, and Redis stream tests
python -m unittest discover -s tests
```

### 5. Run locally
```bash
# Start the Flask web application
python app.py
```
Open `http://localhost:5050` in your browser.

*(Optional)* If running Redis Streams locally, start the decoupled task worker in a separate terminal:
```bash
python task_worker.py
```

---

## 📁 Project Structure

```
medibot/
├── app.py                     # Application factory, route registration, lifecycle cleanup
├── routes/
│   ├── __init__.py            # Blueprint aggregators
│   ├── auth.py                # Authentication, OTP password reset, Google OAuth 2.0
│   ├── chat.py                # Main chat API (/get), session management, /health, /ping
│   └── voice.py               # LiveKit token dispatch & voice chat streaming endpoint
├── research/
│   └── src/
│       ├── clinical_triage.py # Auditable triage matrix, PatientState, red-flag overrides & dosing rules
│       ├── guardrails.py      # Prompt injection tripwires, 911 emergencies, decision-support sanitization
│       ├── intent_classifier.py# Sub-millisecond two-tier hybrid intent classifier (Regex + Groq)
│       ├── helper.py          # Zero-download, zero-RAM deterministic 384D normalized vector generator
│       ├── memory.py          # Asynchronous per-user longitudinal clinical memory extraction
│       └── auth.py            # SQLAlchemy database models (User, ChatSession, Message)
├── services/
│   ├── ai_service.py          # Groq dual-model failover, CustomPineconeRetriever, dynamic prompt builder
│   ├── chat_service.py        # Context window summarization, session titling, background handlers
│   ├── email_service.py       # Resilient multi-provider email (Brevo HTTP API, Resend, SMTP fallback)
│   └── task_dispatcher.py     # Redis Streams publisher, SHA-256 idempotency deduplication, fallback executor
├── task_worker.py             # Decoupled Redis Streams background consumer with retries and DLQ
├── voice_agent.py             # LiveKit voice pipeline (multilingual STT, LLM, Aura-2 TTS)
├── voice_worker.py            # Standalone LiveKit background worker process
├── tests/
│   ├── test_adversarial_triage.py # 8 adversarial flows (late chemo disclosure, infant sepsis, preeclampsia)
│   ├── test_task_queue.py         # 6 Redis Stream, idempotency, retry backoff, and DLQ tests
│   ├── test_guardrails_and_rag.py # 10 guardrail tripwires, emergency detection, RAG pipeline tests
│   └── test_app.py                # 6 auth, session lifecycle, and Brevo/SMTP email tests
├── docs/
│   └── study_notes_and_hld.md # Comprehensive system study notes, HLD architecture & engineering guide
├── scripts/
│   ├── generate_study_hld_diagram.py # HLD architecture diagram generator
│   └── compile_both_documents_pdf.py # Unified documentation compiler
├── requirements.txt           # Minimal, lightweight dependencies (<50MB RAM footprint)
├── gunicorn.conf.py           # Production Gunicorn config supervising Voice & Task workers
└── Procfile                   # Web process definition for cloud deployment
```

---

## 🧪 Comprehensive Automated Test Suite

MediBot is verified through an automated test suite containing **30 exhaustive test cases**:

1. **Adversarial Clinical Triage (`tests/test_adversarial_triage.py`)**:
   - **Late High-Risk Disclosure**: User mentions chemotherapy mid-chat $\to$ immediate Febrile Neutropenia emergency escalation + retroactive correction alert.
   - **Neonatal Fever ($< 3$ months)**: Immediate emergency pediatric escalation with strict OTC medication blocks.
   - **Obstetric Red-Flags**: Severe preeclampsia symptoms during pregnancy $\to$ immediate obstetric emergency triage.
   - **Undisclosed History Dosing**: User requests exact drug mg/pill count $\to$ blocked with clinical safety rationale.
   - **Pediatric Weight-Based Restriction**: Dosing calculations blocked for patients $<12$ years old.
   - **Non-Diagnostic Phrasing**: Diagnostic assertions converted to legally compliant decision-support phrasing.
   - **Single Disclaimer Policy**: Guarantees medical disclaimer is not repetitively spammed across turns.

2. **Decoupled Task Queue & Redis Streams (`tests/test_task_queue.py`)**:
   - **Stream Enqueue**: Verifies durable task publishing to Redis Streams via `XADD`.
   - **Idempotency Deduplication**: Verifies that duplicate tasks (`SET NX`) are dropped before reaching the stream or consuming LLM tokens.
   - **Exponential Backoff**: Verifies that transient failures (e.g., Groq 429 rate limits) trigger exponential backoff retries (up to 3 times).
   - **Dead-Letter Queue (DLQ)**: Verifies unrecoverable tasks are routed to `medical-tasks-dlq`.
   - **Fallback Graceful Execution**: Verifies background tasks execute cleanly when Redis is unreachable.

3. **Guardrails, Emergencies & RAG Pipeline (`tests/test_guardrails_and_rag.py`)**:
   - Prompt injection interception (DAN, jailbreaks, delimiter escape).
   - Instant 911 emergency tripwire (chest pain, stroke symptoms).
   - Grounded RAG retrieval from *The Gale Encyclopedia of Medicine*.
   - Dual-model Groq LLM initialization and template escape safety.

4. **Application & Authentication Flows (`tests/test_app.py`)**:
   - User signup, password strength validation, session creation, and deletion.
   - OTP generation, Brevo HTTP API dispatch, and password reset verification.

### Running the Suite:
```bash
python -m unittest discover -s tests
```
```text
Ran 30 tests in 16.170s

OK
```

---

## 🌐 Production Deployment & Process Supervision

MediBot is pre-configured for deployment on platforms like **Render**:
1. **Web Service Configuration**:
   - Connect your GitHub repository to Render as a **Web Service**.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --config gunicorn.conf.py app:app`
2. **Environment Variables**:
   - Set all mandatory variables: `GROQ_API_KEY`, `PINECONE_API_KEY`, `DEEPGRAM_API_KEY`, `LIVEKIT_*`, `GOOGLE_*`, `SECRET_KEY`.
   - Optional: Provide `REDIS_URL` and `BREVO_API_KEY`.
3. **Master Process Supervision**:
   - `gunicorn.conf.py` runs in `gthread` mode with controlled worker limits (`max_requests=100`, `jitter=25`) to prevent RAM leaks.
   - Gunicorn hooks (`on_starting`, `on_exit`) launch and supervise background threads that manage `voice_worker.py` and `task_worker.py` subprocesses with automatic crash recovery, exponential backoff, and clean termination (`@atexit` / `SIGTERM`).
4. **24/7 Zero-Sleep Uptime Monitoring**:
   - Render free-tier instances sleep after 15 minutes of inactivity. To keep MediBot awake 24/7, configure an external uptime monitor (such as [UptimeRobot](https://uptimerobot.com) or [Cron-job.org](https://cron-job.org)) to ping the public health endpoint every **5 minutes**:
   - **`https://medibot-22m0.onrender.com/health`**

---

## 📄 Medical & Legal Disclaimer

*MediAssist is an AI-powered clinical decision-support and health educational tool grounded in peer-reviewed medical literature. It does not provide definitive medical diagnoses, formulate individualized treatment plans, or write prescriptions. It is not a replacement for professional clinical judgment. Always consult a qualified physician or healthcare provider regarding any acute, urgent, or chronic medical condition.*

---

## 🙋 Author

Built with ❤️ by [Parth Tyagi](https://github.com/parthTyagi-tech).
