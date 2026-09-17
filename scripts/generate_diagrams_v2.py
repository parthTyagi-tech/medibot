"""
Generate high-resolution architecture diagrams for the MediAssist Technical Masterclass v2.
Outputs 3 professional PNG diagrams to ./docs/images/:
  1. 01_distributed_system_architecture.png
  2. 02_rag_evaluation_lifecycle.png
  3. 03_voice_audio_turn_management.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# Executive Dark Theme Palette
BG_COLOR = "#0B0F19"
PANEL_BG = "#111827"
BORDER_COLOR = "#1F2937"
TEXT_WHITE = "#F9FAFB"
TEXT_MUTED = "#9CA3AF"
ACCENT_BLUE = "#38BDF8"
ACCENT_CYAN = "#06B6D4"
ACCENT_GREEN = "#10B981"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_ROSE = "#F43F5E"

def setup_canvas(width=18, height=11, dpi=200):
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax

def draw_card(ax, x, y, w, h, bg=PANEL_BG, border=BORDER_COLOR, lw=1.5, radius=1.5, alpha=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg, edgecolor=border, linewidth=lw, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=ACCENT_CYAN, lw=2.0, style="->", connectionstyle="arc3,rad=0", zorder=4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=connectionstyle,
            shrinkA=4, shrinkB=4
        ),
        zorder=zorder
    )

# ==============================================================================
# DIAGRAM 1: Decoupled Enterprise Distributed System Architecture
# ==============================================================================
def generate_diagram_1():
    fig, ax = setup_canvas(18, 11, dpi=200)

    ax.text(50, 96.5, "MediAssist — Decoupled Enterprise Distributed Architecture",
            ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93.5, "Asynchronous Broker Pipeline, Circuit Breakers, Multi-Tenant Vector Search & Telemetry Observability",
            ha="center", va="center", fontsize=10.5, color=TEXT_MUTED)

    # COLUMN 1: Ingress & Edge (x: 2 to 20)
    draw_card(ax, 2, 10, 18, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(11, 86, "INGRESS & EDGE TIER", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_BLUE)

    draw_card(ax, 3.5, 68, 15, 15, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(11, 78, "Client Applications", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(11, 72.5, "• Next.js / Single-Page App\n• Mobile Native (iOS / Android)\n• LiveKit WebRTC SDK (Opus)\n• EventSource / SSE Client",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 3.5, 48, 15, 16, bg=PANEL_BG, border=ACCENT_BLUE, lw=1.5)
    ax.text(11, 59.5, "Edge & API Gateway", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(11, 53.5, "• Cloudflare / Envoy Gateway\n• TLS Termination (HTTP/3)\n• Token Bucket Rate Limiting\n• WAF & DDoS Shielding",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 3.5, 16, 15, 28, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(11, 38.5, "LiveKit WebRTC Server", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(11, 30.5, "• Selective Forwarding Unit (SFU)\n• Low-Latency Media Routing\n• Adaptive Bitrate & Jitter Buffer\n• DataChannel for Transcripts\n• Dispatches Voice Agent Pods",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # COLUMN 2: API Gateway & Guardrails (x: 23 to 44)
    draw_card(ax, 23, 10, 21, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(33.5, 86, "GATEWAY & GUARDRAIL TIER", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_CYAN)

    draw_card(ax, 24.5, 68, 18, 15, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(33.5, 78, "Stateless API Replicas", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(33.5, 72.5, "• Gunicorn / Uvicorn (K8s / Cloud Run)\n• Autoscale on CPU / Latency\n• Stateless Session Validation (JWT)\n• Zero In-Process Daemon Threads",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 24.5, 48, 18, 16, bg=PANEL_BG, border=ACCENT_RED, lw=1.5)
    ax.text(33.5, 59.5, "Deterministic Guardrails", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_RED)
    ax.text(33.5, 53.5, "• Regex Emergency Interceptor (0ms)\n  (Cardiac arrest, FAST stroke, poison)\n• Prompt Injection & DAN Defense\n• Clinical Triage Risk Matrix\n• Contraindication Dosing Blocks",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 24.5, 16, 18, 28, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(33.5, 38.5, "Durable Event Publisher", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_AMBER)
    ax.text(33.5, 30.5, "• Publishes async tasks to broker\n• Guaranteed At-Least-Once Delivery\n• Idempotency-Key generation\n• Never blocks user HTTP request\n• Eliminates WSGI thread recycling risk",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # COLUMN 3: Decoupled Queue & Workers (x: 47 to 69)
    draw_card(ax, 47, 10, 22, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(58, 86, "DECOUPLED ASYNC WORKERS", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_AMBER)

    draw_card(ax, 48.5, 68, 19, 15, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(58, 78, "Durable Message Broker", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(58, 72.5, "• Redis Streams / GCP Cloud Tasks\n• Celery / SQS with Persistent Acks\n• Dead-Letter Queue (DLQ) for errors\n• Exponential backoff with full jitter",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 48.5, 48, 19, 16, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(58, 59.5, "Memory Extractor Worker", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_GREEN)
    ax.text(58, 53.5, "• Consumes from 'medical-memory' topic\n• Extracts allergies, conditions, vitals\n• Deduplicates facts with LLM prompt\n• Commits update to User.memory in DB\n• Isolated from HTTP lifecycle",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 48.5, 16, 19, 28, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(58, 38.5, "Summarizer & Title Worker", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_PURPLE)
    ax.text(58, 30.5, "• Consumes from 'session-summary' topic\n• Triggered on msg_count >= 6 and % 4\n• Compresses earlier turns into 3 bullets\n• Writes summary to ChatSession in DB\n• Title Worker generates 4-6 word tag",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # COLUMN 4: AI Engine, Storage & Observability (x: 72 to 98)
    draw_card(ax, 72, 53, 26, 37, bg="#0F172A", border="#334155", radius=2)
    ax.text(85, 86, "AI & VECTOR INFERENCE TIER", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_PURPLE)

    draw_card(ax, 73.5, 56, 23, 27, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(85, 77.5, "Groq LPU + Pinecone Cluster", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(85, 68, "• Pinecone: Namespaced Multi-Tenant RAG\n  (Sub-25ms ANN cosine search)\n• Primary LLM: Llama 3.3-70B on Groq LPUs\n• Fallback LLM: GPT-OSS-120B on 429 quota\n• Circuit Breaker (pybreaker / resilience)\n• Sub-300ms Time To First Token (TTFT)",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 72, 10, 26, 40, bg="#0F172A", border="#334155", radius=2)
    ax.text(85, 46, "STORAGE & OBSERVABILITY", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_GREEN)

    draw_card(ax, 73.5, 29, 23, 14, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(85, 38.5, "PostgreSQL HA + PgBouncer", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(85, 33.5, "• Managed Cloud SQL / RDS with Replicas\n• PgBouncer Connection Pooling\n• Users, ChatSessions, Messages, Memory",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 73.5, 13, 23, 13, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(85, 21.5, "OpenTelemetry & Langfuse", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(85, 17, "• Traces: Latency, Prompt/Completion Tokens\n• Cosine Similarity & Vector Drift Tracking\n• Real-Time Medico-Legal Audit Trails",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # ARROWS
    draw_arrow(ax, 18.5, 75, 24.5, 75, color=ACCENT_CYAN, lw=2) # Client -> Gateway
    draw_arrow(ax, 18.5, 30, 24.5, 70, color=ACCENT_PURPLE, lw=1.8, connectionstyle="arc3,rad=-0.15") # LiveKit -> Gateway /voice_chat
    draw_arrow(ax, 33.5, 68, 33.5, 64, color=ACCENT_RED, lw=1.5) # Gateway -> Guardrails
    draw_arrow(ax, 33.5, 48, 33.5, 44, color=ACCENT_AMBER, lw=1.5) # Guardrails -> Event Publisher
    
    # Event Publisher -> Broker -> Workers
    draw_arrow(ax, 42.5, 30, 48.5, 75, color=ACCENT_AMBER, lw=2, connectionstyle="arc3,rad=-0.15") # Publisher to Broker
    draw_arrow(ax, 58, 68, 58, 64, color=ACCENT_AMBER, lw=1.5) # Broker -> Worker 1
    draw_arrow(ax, 58, 48, 58, 44, color=ACCENT_AMBER, lw=1.5) # Worker 1 -> Worker 2
    
    # Gateway & Workers to DB and AI Tier
    draw_arrow(ax, 42.5, 75, 73.5, 75, color=ACCENT_PURPLE, lw=2) # Gateway -> Groq/Pinecone
    draw_arrow(ax, 67.5, 56, 73.5, 38, color=ACCENT_GREEN, lw=1.8) # Workers -> PostgreSQL
    draw_arrow(ax, 42.5, 68, 73.5, 20, color=ACCENT_CYAN, lw=1.5, connectionstyle="arc3,rad=0.15") # Gateway -> OTel

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_distributed_system_architecture.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ==============================================================================
# DIAGRAM 2: Production RAG Lifecycle, Evaluation & Observability
# ==============================================================================
def generate_diagram_2():
    fig, ax = setup_canvas(18, 11, dpi=200)

    ax.text(50, 96.5, "MediAssist — Production RAG Lifecycle & Continuous Evaluation",
            ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93.5, "Blue/Green Zero-Downtime Indexing, Ragas Metric Quality Gates, and Runtime Drift Observability",
            ha="center", va="center", fontsize=10.5, color=TEXT_MUTED)

    # 3 HORIZONTAL / PIPELINE SECTIONS
    # SECTION 1: Blue/Green Indexing & Pipeline (Left, x: 2 to 32)
    draw_card(ax, 2, 10, 30, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(17, 86, "1. BLUE/GREEN INDEXING PIPELINE", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_BLUE)

    draw_card(ax, 4, 69, 26, 14, bg=PANEL_BG, border=ACCENT_BLUE, lw=1.5)
    ax.text(17, 78, "Medical Knowledge Corpus", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(17, 73, "• Gale Encyclopedia of Medicine (v1-5)\n• CDC / WHO Clinical Protocols\n• PDF extraction via PyPDFLoader",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 4, 51, 26, 15, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(17, 60.5, "Semantic Chunking (2500/50)", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(17, 55.5, "• 2500 chars: Preserves complete clinical\n  monographs (Etiology, Signs, Rx)\n• 50 char overlap: Prevents splitting\n  compound drugs & lab cutoff ranges",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 4, 33, 26, 15, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(17, 42.5, "Deterministic Embeddings", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(17, 37.5, "• 384-dimensional normalized vectors\n• Zero-RAM footprint, 0.001ms gen\n• L2-normalized: Cosine == Dot Product\n• In-memory 128-LRU query vector cache",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 4, 13, 26, 17, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(17, 23.5, "Blue/Green Vector Cutover", ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN)
    ax.text(17, 17.5, "• Active Index (Blue): Serving live traffic\n• Candidate Index (Green): New embeddings\n• Batched upserting (100 vectors / req)\n• Zero downtime atomic traffic flip",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # Connecting arrows Section 1
    draw_arrow(ax, 17, 69, 17, 66, color=ACCENT_BLUE, lw=1.8)
    draw_arrow(ax, 17, 51, 17, 48, color=ACCENT_CYAN, lw=1.8)
    draw_arrow(ax, 17, 33, 17, 30, color=ACCENT_PURPLE, lw=1.8)

    # SECTION 2: Automated Offline Ragas Evaluation (Middle, x: 35 to 65)
    draw_card(ax, 35, 10, 30, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(50, 86, "2. AUTOMATED RAGAS CI/CD GATE", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_AMBER)

    draw_card(ax, 37, 69, 26, 14, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(50, 78, "Golden Clinical Test Suite", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 73, "• 500+ curated doctor-reviewed pairs\n• Emergency red flags & rare pathologies\n• Negative controls (non-medical / traps)",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 37, 46, 26, 20, bg=PANEL_BG, border=ACCENT_ROSE, lw=1.5)
    ax.text(50, 59.5, "Ragas Metric Triad Evaluation", ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT_ROSE)
    ax.text(50, 52, "• Faithfulness (Groundedness):\n  Entailment of claims from Gale text\n• Context Recall:\n  Retrieval of all required clinical facts\n• Answer Relevance:\n  Direct clinical triage without bloat",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 37, 24, 26, 19, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(50, 37, "CI/CD Promotion Gate", ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN)
    ax.text(50, 30, "• Threshold Check:\n  Faithfulness >= 0.94\n  Context Recall >= 0.90\n  Answer Relevance >= 0.92\n• Pass -> Cut traffic to Green index\n• Fail -> Block deploy & alert triage team",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 37, 13, 26, 8, bg=PANEL_BG, border="#475569", lw=1)
    ax.text(50, 17, "Target: Zero Hallucination Deployments", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)

    # Connecting arrows Section 2
    draw_arrow(ax, 50, 69, 50, 66, color=ACCENT_AMBER, lw=1.8)
    draw_arrow(ax, 50, 46, 50, 43, color=ACCENT_ROSE, lw=1.8)

    # SECTION 3: Live Observability & Drift Detection (Right, x: 68 to 98)
    draw_card(ax, 68, 10, 30, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(83, 86, "3. RUNTIME DRIFT & OBSERVABILITY", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_CYAN)

    draw_card(ax, 70, 69, 26, 14, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(83, 78, "OpenTelemetry Span Tracing", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(83, 73, "• Span: retriever.vector_lookup\n• Span: groq.llm_generation\n• Captures latency, prompt tokens, cost",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 70, 48, 26, 18, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(83, 59.5, "Langfuse Clinical Logging", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(83, 53, "• Log query text + retrieved chunk IDs\n• Cosine similarity distribution logging\n• User thumb-up / thumb-down signals\n• Real-time token-cost-per-turn audit",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 70, 24, 26, 21, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(83, 38, "Vector Drift Detection", ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT_AMBER)
    ax.text(83, 30.5, "• Centroid Drift Tracking:\n  Monitors query embedding centroid shift\n• Similarity Degradation Alert:\n  Fires alert if Top-1 cosine score drops\n  below 0.72 rolling average (outdated index)\n• Automated pipeline re-trigger",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 70, 13, 26, 8, bg=PANEL_BG, border="#475569", lw=1)
    ax.text(83, 17, "Full Medico-Legal Traceability", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)

    # Connecting arrows Section 3
    draw_arrow(ax, 83, 69, 83, 66, color=ACCENT_CYAN, lw=1.8)
    draw_arrow(ax, 83, 48, 83, 45, color=ACCENT_PURPLE, lw=1.8)

    # Cross connections across sections
    draw_arrow(ax, 30, 21, 37, 21, color=ACCENT_GREEN, lw=2) # Green index to CI/CD gate
    draw_arrow(ax, 63, 21, 70, 75, color=ACCENT_CYAN, lw=2, connectionstyle="arc3,rad=-0.15") # Promoted index to Runtime

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "02_rag_evaluation_lifecycle.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ==============================================================================
# DIAGRAM 3: Voice Audio Turn Management & Barge-In Cancellation
# ==============================================================================
def generate_diagram_3():
    fig, ax = setup_canvas(18, 11, dpi=200)

    ax.text(50, 96.5, "MediAssist — Voice Audio Turn Management & Barge-In Architecture",
            ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93.5, "Full-Duplex WebRTC Loop, Speculative Token Streaming & Synchronized Database State Truncation",
            ha="center", va="center", fontsize=10.5, color=TEXT_MUTED)

    # TOP HALF: Speculative Pipelined Streaming (x: 2 to 98, y: 52 to 89)
    draw_card(ax, 2, 52, 96, 38, bg="#0F172A", border=ACCENT_GREEN, radius=2)
    ax.text(50, 85, "PIPELINED SPECULATIVE TOKEN STREAMING (OVERLAPPING TTS & LLM INFERENCE)", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_GREEN)

    # Subcards showing stages
    stages = [
        ("1. Audio Ingress", "• 20ms Opus frames\n• WebRTC UDP (RTP)\n• Echo Cancellation", ACCENT_BLUE, 4),
        ("2. Silero VAD", "• Prewarmed in memory\n• 250ms speech min\n• 500ms silence endpoint", ACCENT_CYAN, 20),
        ("3. Deepgram STT", "• Streaming WebSocket\n• Multilingual Nova-2\n• <120ms latency", ACCENT_AMBER, 36),
        ("4. Groq LPU LLM", "• Llama 3.3-70B\n• Yields first 6 tokens\n• TTFT < 280ms", ACCENT_PURPLE, 52),
        ("5. Deepgram TTS", "• Speculative chunk synth\n• Starts audio on 1st clause\n• TTFB < 150ms", ACCENT_ROSE, 68),
        ("6. Speaker Out", "• Audio play to client\n• User hears voice in\n• ~650ms total turn", ACCENT_GREEN, 84)
    ]
    for title, text, color, x_pos in stages:
        draw_card(ax, x_pos, 55, 14, 25, bg=PANEL_BG, border=color, lw=1.5)
        ax.text(x_pos + 7, 75, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=TEXT_WHITE)
        ax.text(x_pos + 7, 65, text, ha="center", va="center", fontsize=7.5, color=TEXT_MUTED)
        if x_pos < 84:
            draw_arrow(ax, x_pos + 14, 67, x_pos + 19.5, 67, color=color, lw=2)

    # BOTTOM HALF: User Barge-In & State Truncation (x: 2 to 98, y: 8 to 48)
    draw_card(ax, 2, 8, 96, 41, bg="#0F172A", border=ACCENT_RED, radius=2)
    ax.text(50, 44, "CRITICAL CONCURRENCY: BARGE-IN INTERRUPTION & STATE TRUNCATION PROTOCOL", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_RED)

    # Step 1
    draw_card(ax, 5, 12, 26, 27, bg=PANEL_BG, border=ACCENT_RED, lw=1.5)
    ax.text(18, 34, "1. Mid-Speech Interruption", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(18, 24, "• Agent is streaming audio Turn N\n• Patient speaks: 'Wait, doctor...'\n• Silero VAD detects speech energy\n  exceeding 250ms threshold\n• Fires barge-in interruption signal",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # Step 2
    draw_card(ax, 37, 12, 26, 27, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(50, 34, "2. RTP Silence & Cancellation", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 24, "• LiveKit server sends RTP silence frame\n• Cancels browser audio playback (<100ms)\n• TTS audio stream cancelled immediately\n• Prevents agent talking over patient\n• Records audio playback offset (ms)",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # Step 3
    draw_card(ax, 69, 12, 26, 27, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(82, 34, "3. Synchronized State Purge", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(82, 24, "• DB Truncation: Message.content in\n  PostgreSQL is clipped to words spoken\n• Unspoken projected text is purged\n• Prevents phantom clinical advice\n  from polluting sliding context window\n• Agent awaits prompt; raises StopResponse",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_arrow(ax, 31, 25, 37, 25, color=ACCENT_RED, lw=2.2)
    draw_arrow(ax, 63, 25, 69, 25, color=ACCENT_AMBER, lw=2.2)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "03_voice_audio_turn_management.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    print("Generating v2 diagrams...")
    generate_diagram_1()
    generate_diagram_2()
    generate_diagram_3()
    print("All v2 diagrams successfully generated in ./docs/images/")
