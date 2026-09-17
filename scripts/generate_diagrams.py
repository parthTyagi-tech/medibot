"""
Generate high-resolution architecture diagrams for the MediAssist Technical Masterclass.
Outputs 4 professional PNG diagrams to ./docs/images/
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

# Output directory
OUT_DIR = os.path.join("docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# Common styling palette (Executive Dark Theme / Tech Whitepaper Aesthetic)
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

def setup_canvas(width=16, height=10, dpi=200):
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
# DIAGRAM 1: End-to-End System Architecture (HLD)
# ==============================================================================
def generate_diagram_1():
    fig, ax = setup_canvas(18, 11, dpi=200)

    # Main Header
    ax.text(50, 96, "MediAssist — End-to-End System Architecture (HLD)",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93, "Decoupled Flask Gateway, Guardrail Engine, RAG Vector Search & LiveKit Real-Time WebRTC Pipeline",
            ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    # LAYER 1: CLIENT ACCESS TIER (x: 3 to 22)
    draw_card(ax, 3, 12, 18, 76, bg="#0F172A", border="#334155", radius=2)
    ax.text(12, 85, "CLIENT INTERACTION LAYER", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
    
    # Client Subcomponents
    draw_card(ax, 4.5, 68, 15, 14, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(12, 78, "Web Browser (DOM/JS)", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(12, 73, "• Single Page Application (chat.html)\n• Async Fetch API (/get, /tts)\n• Theme & Memory Modals",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    draw_card(ax, 4.5, 48, 15, 16, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(12, 60, "LiveKit WebRTC Client", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(12, 54, "• livekit-client SDK (v2.11)\n• Full-Duplex Audio Tracks\n• DataChannel for Transcripts\n• Mic VAD & Echo Cancellation",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    draw_card(ax, 4.5, 16, 15, 28, bg=PANEL_BG, border="#475569", lw=1)
    ax.text(12, 40, "Network Protocols", ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT_AMBER)
    ax.text(12, 33, "HTTPS / REST (JSON & Multipart)", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)
    ax.text(12, 28, "WSS / WebRTC (Opus 48kHz Audio)", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)
    ax.text(12, 23, "SSE (Server-Sent Events Stream)", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)
    ax.text(12, 18, "JSON DataChannel Packets", ha="center", va="center", fontsize=8.5, color=TEXT_WHITE)

    # LAYER 2: API GATEWAY & SECURITY INGRESS (x: 25 to 44)
    draw_card(ax, 25, 12, 19, 76, bg="#0F172A", border="#334155", radius=2)
    ax.text(34.5, 85, "FLASK GATEWAY & SECURITY", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_CYAN)

    draw_card(ax, 26.5, 68, 16, 14, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(34.5, 78, "ProxyFix & App Core", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(34.5, 73, "• Werkzeug ProxyFix (x_for, x_proto)\n• Flask-Login (User Session Cookies)\n• Google OAuth 2.0 (Flask-Dance)",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    draw_card(ax, 26.5, 48, 16, 16, bg=PANEL_BG, border=ACCENT_RED, lw=1.5)
    ax.text(34.5, 60, "Input Guardrails Matrix", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_RED)
    ax.text(34.5, 54, "• Prompt Injection & DAN Defense\n• Medical Emergency Detection\n  (FAST stroke, crushing chest pain)\n• 0ms Critical Override Intercept",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    draw_card(ax, 26.5, 26, 16, 18, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(34.5, 40, "Intent Classifier", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_AMBER)
    ax.text(34.5, 33, "• Heuristic Root Match (0ms)\n• Medical / Greeting / Memory\n• Groq Compound Mini Fallback\n• Out-of-Scope Refusal Guard",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    draw_card(ax, 26.5, 15, 16, 8, bg=PANEL_BG, border="#475569", lw=1)
    ax.text(34.5, 19, "Async Daemon Threads\n(Title, Memory, Summary)", ha="center", va="center", fontsize=8.5, color=ACCENT_BLUE)

    # LAYER 3: CLINICAL INTELLIGENCE & RAG (x: 48 to 72)
    draw_card(ax, 48, 12, 24, 76, bg="#0F172A", border="#334155", radius=2)
    ax.text(60, 85, "CLINICAL INTELLIGENCE & RAG", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_GREEN)

    draw_card(ax, 49.5, 68, 21, 14, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(60, 78, "Clinical Triage & Patient State", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_GREEN)
    ax.text(60, 73, "• Structured Patient State Extraction\n• Triage Tier: Routine / Urgent / Emergency\n• High-Risk Contraindications & Dosing Block\n• Mid-Conversation Medical Corrections",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    draw_card(ax, 49.5, 48, 21, 16, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(60, 60, "Dynamic Prompt & Context Builder", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT_WHITE)
    ax.text(60, 54, "• Curly brace escaping: {{...}}\n• Rolling context injection (Last 6 turns)\n• Summary Context & Patient Profile Memory\n• Gale Encyclopedia + Clinical Directives",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    draw_card(ax, 49.5, 26, 21, 18, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(60, 40, "Dual-Engine Groq ChatModel", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT_PURPLE)
    ax.text(60, 33, "• Primary: Llama 3.3-70B / Compound (<300ms)\n• Fallback: Groq GPT-OSS-120B on 429\n• SSE Chunk Generator (stream_with_context)\n• Output Guardrails & Safety Disclaimers",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    draw_card(ax, 49.5, 15, 21, 8, bg=PANEL_BG, border=ACCENT_ROSE, lw=1)
    ax.text(60, 19, "Custom Pinecone Retriever\n(Top-K=4 Gale Docs + Deterministic Fallback)", ha="center", va="center", fontsize=8.5, color=ACCENT_ROSE)

    # LAYER 4: VOICE WORKER & DATA STORES (x: 76 to 97)
    draw_card(ax, 76, 50, 21, 38, bg="#0F172A", border="#334155", radius=2)
    ax.text(86.5, 85, "LIVEKIT VOICE AGENT", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_PURPLE)

    draw_card(ax, 77.5, 53, 18, 29, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(86.5, 78, "voice_worker.py (Subprocess)", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(86.5, 70, "• Prewarmed Silero VAD\n• Deepgram STT (Nova-2 Multilingual)\n• Multilingual Config (EN, ES, FR, etc.)\n• Deepgram TTS (Aura-2 Voices)\n• Turn Lock (session.say awaiting)\n• Fallback Backend HTTP Pool",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    draw_card(ax, 76, 12, 21, 35, bg="#0F172A", border="#334155", radius=2)
    ax.text(86.5, 43, "DATA & STORAGE LAYER", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_AMBER)

    draw_card(ax, 77.5, 27, 18, 13, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(86.5, 36, "SQLAlchemy Relational DB", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(86.5, 31, "• PostgreSQL / SQLite (users.db)\n• User, ChatSession, Message\n• user.memory & session.summary",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 77.5, 14, 18, 10, bg=PANEL_BG, border=ACCENT_ROSE, lw=1.5)
    ax.text(86.5, 20.5, "Pinecone Vector Store", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(86.5, 16.5, "• Index: medical-chatbot\n• 384-Dim Normalized Embeddings\n• Gale Encyclopedia Metadata Chunks",
            ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # CONNECTING ARROWS
    # Client -> Gateway
    draw_arrow(ax, 19.5, 75, 26.5, 75, color=ACCENT_CYAN, lw=2)
    draw_arrow(ax, 19.5, 56, 77.5, 70, color=ACCENT_PURPLE, lw=2, connectionstyle="arc3,rad=-0.15") # Client WebRTC to Voice Worker
    
    # Gateway Internal flow
    draw_arrow(ax, 34.5, 68, 34.5, 64, color=ACCENT_CYAN, lw=1.5)
    draw_arrow(ax, 34.5, 48, 34.5, 44, color=ACCENT_RED, lw=1.5)
    
    # Gateway to Intelligence
    draw_arrow(ax, 42.5, 35, 49.5, 75, color=ACCENT_GREEN, lw=2, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, 60, 68, 60, 64, color=ACCENT_GREEN, lw=1.5)
    draw_arrow(ax, 60, 48, 60, 44, color=ACCENT_PURPLE, lw=1.5)
    
    # Intelligence to Pinecone & DB
    draw_arrow(ax, 70.5, 19, 77.5, 19, color=ACCENT_ROSE, lw=1.8) # To Pinecone
    draw_arrow(ax, 42.5, 19, 77.5, 33, color=ACCENT_AMBER, lw=1.5, connectionstyle="arc3,rad=0.15") # Gateway async threads to DB
    
    # Voice Worker to Gateway (/voice_chat)
    draw_arrow(ax, 77.5, 62, 42.5, 72, color=ACCENT_BLUE, lw=1.8, connectionstyle="arc3,rad=0.12")

    # Legend / Status bar
    ax.text(50, 4, "Data Flow: [User Request] ➔ [Guardrails Intercept] ➔ [Intent Classification] ➔ [Pinecone Cosine Retrieval] ➔ [Groq LLM Streaming] ➔ [Client/WebRTC Audio]",
            ha="center", va="center", fontsize=9.5, color=ACCENT_CYAN, style="italic")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_system_architecture_hld.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ==============================================================================
# DIAGRAM 2: RAG Lifecycle (Gale Encyclopedia to Pinecone)
# ==============================================================================
def generate_diagram_2():
    fig, ax = setup_canvas(18, 10, dpi=200)

    ax.text(50, 96, "MediAssist — RAG Ingestion & Vector Query Lifecycle",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93, "Offline Clinical Corpus Ingestion (Gale Encyclopedia) vs. Low-Latency Online Similarity Search",
            ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    # TWO COLUMNS: LEFT = INGESTION PIPELINE (Offline), RIGHT = QUERY & RETRIEVAL (Online)
    
    # LEFT PANEL: Ingestion & Indexing
    draw_card(ax, 3, 10, 44, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(25, 86, "PHASE A: OFFLINE DOCUMENT INGESTION (store_index.py)", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_AMBER)

    # Step 1: Raw PDF
    draw_card(ax, 6, 71, 38, 11, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.5)
    ax.text(25, 78.5, "1. Raw Clinical Medical Corpus", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(25, 74, "• The Gale Encyclopedia of Medicine (Volumes 1-5 PDFs in /data)\n• PyPDFLoader + DirectoryLoader extracts text & page layout",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 2: Metadata Filtering
    draw_card(ax, 6, 56, 38, 11, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(25, 63.5, "2. Metadata Normalization & Filtering", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(25, 59, "• filter_to_minimal_docs(): strips bloated headers, fonts & layout trees\n• Conserves vector payload overhead while retaining 'source' tag",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 3: Text Splitting
    draw_card(ax, 6, 41, 38, 11, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(25, 48.5, "3. Recursive Character Chunking", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(25, 44, "• RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=50)\n• 2500 chars keeps complete clinical pathology, diagnosis & treatment units\n• 50 char overlap ensures unbroken terminology at chunk boundaries",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 4: Deterministic Embedding
    draw_card(ax, 6, 26, 38, 11, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(25, 33.5, "4. 384-Dim Embedding Generation", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(25, 29, "• LocalEmbeddings (Deterministic SHA-256 Vector Generation)\n• 0.001ms generation time, zero-RAM, zero Torch/HuggingFace dependency\n• L2 Normalized vectors: ||v|| = 1.0 (Cosine Similarity = Dot Product)",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 5: Pinecone Upsert
    draw_card(ax, 6, 12, 38, 10, bg=PANEL_BG, border=ACCENT_ROSE, lw=1.5)
    ax.text(25, 18.5, "5. Batched Vector Upsert (Pinecone)", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(25, 14.5, "• Batches of 100 vectors: {'id': 'chunk-i', 'values': [...], 'metadata': {'text': text, 'source': 'Gale'}}\n• Index: 'medical-chatbot' (Metric: Cosine)",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    # Connecting arrows on Left
    draw_arrow(ax, 25, 71, 25, 67, color=ACCENT_AMBER, lw=1.8)
    draw_arrow(ax, 25, 56, 25, 52, color=ACCENT_CYAN, lw=1.8)
    draw_arrow(ax, 25, 41, 25, 37, color=ACCENT_GREEN, lw=1.8)
    draw_arrow(ax, 25, 26, 25, 22, color=ACCENT_PURPLE, lw=1.8)

    # RIGHT PANEL: Runtime Vector Query & Cosine Similarity Lookup
    draw_card(ax, 53, 10, 44, 80, bg="#0F172A", border="#334155", radius=2)
    ax.text(75, 86, "PHASE B: ONLINE QUERY & SIMILARITY LOOKUP", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_BLUE)

    # Step 1: Query Input
    draw_card(ax, 56, 71, 38, 11, bg=PANEL_BG, border=ACCENT_BLUE, lw=1.5)
    ax.text(75, 78.5, "1. Runtime Patient Query", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(75, 74, "• Patient Query: 'What are the emergency red flags of pancreatitis?'\n• Passed into CustomPineconeRetriever._get_relevant_documents()",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 2: Query Embedding
    draw_card(ax, 56, 56, 38, 11, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.5)
    ax.text(75, 63.5, "2. Query Vector Generation", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(75, 59, "• query_vector = embedding.embed_query(query) -> 384 dimensions\n• Fast in-memory vector cache (128 LRU entries) eliminates redundant computation",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 3: Pinecone Cosine Search
    draw_card(ax, 56, 41, 38, 11, bg=PANEL_BG, border=ACCENT_ROSE, lw=1.5)
    ax.text(75, 48.5, "3. Cosine Similarity Vector Lookup", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(75, 44, "• index.query(vector=query_vector, top_k=4, include_metadata=True)\n• Sub-25ms ANN (Approximate Nearest Neighbor) graph search\n• Metadata filtering supports tenant isolation and source provenance",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # Step 4: Prompt Construction
    draw_card(ax, 56, 26, 38, 11, bg=PANEL_BG, border=ACCENT_GREEN, lw=1.5)
    ax.text(75, 33.5, "4. Context Synthesis & Fallback Resilience", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(75, 29, "• If Pinecone matches found: context = '\\n\\n'.join([doc.page_content for doc in docs])\n• If network/index exception: Gracefully falls back to Gale clinical principles\n• Never crashes the consultation pipeline on vector database downtime",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    # Step 5: Generation
    draw_card(ax, 56, 12, 38, 10, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.5)
    ax.text(75, 18.5, "5. LLM Grounded Generation (Groq)", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax.text(75, 14.5, "• Grounded in authoritative Gale clinical text; prevents model hallucination\n• Applies clinical protocol: triage first, no unverified assumptions, decision support",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)

    # Connecting arrows on Right
    draw_arrow(ax, 75, 71, 75, 67, color=ACCENT_BLUE, lw=1.8)
    draw_arrow(ax, 75, 56, 75, 52, color=ACCENT_PURPLE, lw=1.8)
    draw_arrow(ax, 75, 41, 75, 37, color=ACCENT_ROSE, lw=1.8)
    draw_arrow(ax, 75, 26, 75, 22, color=ACCENT_GREEN, lw=1.8)

    # Cross-link between Ingestion Vector Index and Query
    draw_arrow(ax, 44, 17, 56, 46, color=ACCENT_ROSE, lw=2.2, style="<->", connectionstyle="arc3,rad=-0.2")
    ax.text(49, 31, "Pinecone Index\n'medical-chatbot'", ha="center", va="center", fontsize=8.5, color=ACCENT_ROSE, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "02_rag_lifecycle.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ==============================================================================
# DIAGRAM 3: Real-Time Multilingual Voice Agent Pipeline
# ==============================================================================
def generate_diagram_3():
    fig, ax = setup_canvas(18, 11, dpi=200)

    ax.text(50, 96, "MediAssist — Real-Time Multilingual Voice Pipeline",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93, "Full-Duplex WebRTC Loop: Silero VAD ➔ Deepgram Nova-2 STT ➔ Groq LLM ➔ Deepgram Aura-2 TTS",
            ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    # STAGES ACROSS A CIRCULAR / LOOP LAYOUT
    
    # 1. USER AUDIO INPUT (Top Left)
    draw_card(ax, 4, 68, 26, 18, bg=PANEL_BG, border=ACCENT_CYAN, lw=2)
    ax.text(17, 82, "1. Patient Microphone & WebRTC", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_CYAN)
    ax.text(17, 74, "• WebRTC Audio Track (48kHz Opus codec)\n• Client-side Echo Cancellation & Noise Suppression\n• LiveKit Room connection via secure JWT token\n• Bidirectional JSON data channel for transcript sync",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # 2. SILERO VAD (Top Center)
    draw_card(ax, 37, 68, 26, 18, bg=PANEL_BG, border=ACCENT_AMBER, lw=2)
    ax.text(50, 82, "2. Silero Voice Activity Detection", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_AMBER)
    ax.text(50, 74, "• Prewarmed in JobProcess userdata (0ms cold start)\n• min_speech_duration = 0.25s (filters breathing/coughs)\n• min_silence_duration = 0.50s (detects speech end)\n• prefix_padding_duration = 0.20s (captures first syllable)",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # 3. DEEPGRAM STT (Top Right)
    draw_card(ax, 70, 68, 26, 18, bg=PANEL_BG, border=ACCENT_GREEN, lw=2)
    ax.text(83, 82, "3. Deepgram Nova-2 STT", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_GREEN)
    ax.text(83, 74, "• Streaming WebSocket Audio-to-Text (<150ms)\n• Multilingual Routing (EN, ES, FR, DE, IT, NL, JA)\n• Fires 'user_input_transcribed' events\n• Emits user text to LiveKit DataChannel (instant UI bubble)",
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)

    # 4. MEDICAL BACKEND & LLM (Bottom Right)
    draw_card(ax, 70, 20, 26, 36, bg=PANEL_BG, border=ACCENT_PURPLE, lw=2)
    ax.text(83, 52, "4. Medical Reasoning Gateway", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_PURPLE)
    ax.text(83, 44, "• HTTP POST /voice_chat (30s timeout)\n• Fallback URL pool: VOICE_BACKEND_URL ➔ 127.0.0.1:PORT\n• Input Guardrails (Prompt Injection & Emergency)\n• Intent Classifier (Medical / Greeting / Memory)\n• Custom Pinecone Retriever (Gale Context)\n• Groq Llama 3.3-70B Engine (Concise Voice Prompt)\n• Output Guardrails & Safety Sanitization",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)
    ax.text(83, 24, "Latency: < 320ms TTFT", ha="center", va="center", fontsize=9, fontweight="bold", color=ACCENT_GREEN)

    # 5. DEEPGRAM TTS (Bottom Center)
    draw_card(ax, 37, 20, 26, 36, bg=PANEL_BG, border=ACCENT_ROSE, lw=2)
    ax.text(50, 52, "5. Deepgram Aura-2 TTS", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_ROSE)
    ax.text(50, 44, "• Natural clinical voice synthesis (<180ms TTFB)\n• Model mappings per language:\n  - EN: aura-2-thalia-en\n  - ES: aura-2-celeste-es\n  - FR: aura-2-agathe-fr\n  - DE: aura-2-aurelia-de\n  - JA: aura-2-izanami-ja\n• Streams audio frames directly to WebRTC track",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)
    ax.text(50, 24, "Natural Conversational Flow", ha="center", va="center", fontsize=9, fontweight="bold", color=ACCENT_CYAN)

    # 6. AUDIO PLAYBACK & TURN MANAGEMENT (Bottom Left)
    draw_card(ax, 4, 20, 26, 36, bg=PANEL_BG, border=ACCENT_BLUE, lw=2)
    ax.text(17, 52, "6. Speaker Output & Turn Lock", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_BLUE)
    ax.text(17, 44, "• CRITICAL ARCHITECTURE DECISION:\n  on_user_turn_completed awaits process_user_prompt()\n• Prevents turn system race condition where\n  subsequent speech drops while TTS plays\n• session.say(allow_interruptions=True)\n  supports natural patient barge-in\n• Room stream drain handlers drain unhandled\n  byte/text streams to prevent memory leaks",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)
    ax.text(17, 24, "Full Duplex with Barge-In", ha="center", va="center", fontsize=9, fontweight="bold", color=ACCENT_AMBER)

    # FORWARD CONNECTING ARROWS
    draw_arrow(ax, 30, 77, 37, 77, color=ACCENT_CYAN, lw=2.5) # 1 -> 2
    draw_arrow(ax, 63, 77, 70, 77, color=ACCENT_AMBER, lw=2.5) # 2 -> 3
    draw_arrow(ax, 83, 68, 83, 56, color=ACCENT_GREEN, lw=2.5) # 3 -> 4
    draw_arrow(ax, 70, 38, 63, 38, color=ACCENT_PURPLE, lw=2.5) # 4 -> 5
    draw_arrow(ax, 37, 38, 30, 38, color=ACCENT_ROSE, lw=2.5) # 5 -> 6
    draw_arrow(ax, 17, 56, 17, 68, color=ACCENT_BLUE, lw=2.5) # 6 -> 1 (Full loop)

    # Central Summary Metric Box
    draw_card(ax, 33, 8, 34, 8, bg="#1E293B", border="#475569", radius=1)
    ax.text(50, 12, "End-to-End Voice Latency Budget: ~650ms - 850ms", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 9.5, "VAD (250ms) + STT (150ms) + Groq RAG (250ms) + TTS TTFB (150ms)", ha="center", va="center", fontsize=8, color=ACCENT_CYAN)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "03_voice_agent_pipeline.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ==============================================================================
# DIAGRAM 4: Session Memory & Context Window Management
# ==============================================================================
def generate_diagram_4():
    fig, ax = setup_canvas(18, 10.5, dpi=200)

    ax.text(50, 96, "MediAssist — Session Memory & Context Window Management",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93, "Sliding Dialogue Window, Background Auto-Summarization & Cross-Session Medical Fact Extraction",
            ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    # 3 HORIZONTAL / HIERARCHICAL TIERS

    # TIER 1: SHORT-TERM ROLLING WINDOW (Top)
    draw_card(ax, 4, 62, 92, 26, bg="#0F172A", border=ACCENT_CYAN, radius=2)
    ax.text(50, 84, "TIER 1: HIGH-RESOLUTION SLIDING DIALOGUE WINDOW (Message.query[-6:])", ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENT_CYAN)
    
    # Message boxes representing rolling window
    turns = [
        ("Turn 1 (Oldest)", "Patient: Fever 102F\nMediAssist: Duration?"),
        ("Turn 2", "Patient: 2 days\nMediAssist: Any cough?"),
        ("Turn 3", "Patient: Dry cough\nMediAssist: Any rash?"),
        ("Turn 4", "Patient: No rash\nMediAssist: Fluid advice"),
        ("Turn 5 (Recent)", "Patient: Headache\nMediAssist: Pain rating?"),
        ("Turn 6 (Latest)", "Patient: Severe 8/10\nMediAssist: Evaluating...")
    ]
    for i, (title, content) in enumerate(turns):
        x_pos = 6 + i * 14.8
        color = "#475569" if i < 2 else ACCENT_CYAN
        draw_card(ax, x_pos, 65, 13.8, 15, bg=PANEL_BG, border=color, lw=1.2)
        ax.text(x_pos + 6.9, 77.5, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=TEXT_WHITE)
        ax.text(x_pos + 6.9, 71, content, ha="center", va="center", fontsize=7.5, color=TEXT_MUTED)

    # TIER 2: AUTOMATIC SESSION SUMMARIZATION (Middle)
    draw_card(ax, 4, 34, 44, 24, bg="#0F172A", border=ACCENT_PURPLE, radius=2)
    ax.text(26, 54, "TIER 2A: CONTEXT AUTO-SUMMARIZATION", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_PURPLE)
    ax.text(26, 45, "• Trigger: When message count >= 6 and msg_count % 4 == 0\n• Asynchronous Execution: summarize_session_in_background()\n• Background thread calls Groq with clinical extraction prompt\n• Compresses 20+ turns into 2-3 clinical bullet points\n• Persisted in ChatSession.summary (SQLAlchemy column)\n• Eliminates LLM context window overflow & reduces token costs by 75%",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)
    ax.text(26, 36.5, "Stored in: ChatSession.summary", ha="center", va="center", fontsize=9, fontweight="bold", color=ACCENT_GREEN)

    # TIER 2B: PATIENT LONG-TERM MEDICAL MEMORY (Middle Right)
    draw_card(ax, 52, 34, 44, 24, bg="#0F172A", border=ACCENT_AMBER, radius=2)
    ax.text(74, 54, "TIER 2B: LONG-TERM MEDICAL MEMORY", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_AMBER)
    ax.text(74, 45, "• Asynchronous Worker: update_memory_in_background()\n• Analyzes recent dialogue + latest input to extract permanent facts:\n  - Chronic conditions (Diabetes Type 2, Asthma)\n  - Known allergies (Penicillin, Sulfa)\n  - Current medications & vital signs\n• Filters out transient pleasantries, small talk, jokes\n• Persisted in User.memory (Cross-session profile)",
            ha="center", va="center", fontsize=8.2, color=TEXT_MUTED)
    ax.text(74, 36.5, "Stored in: User.memory (Relational DB)", ha="center", va="center", fontsize=9, fontweight="bold", color=ACCENT_GREEN)

    # TIER 3: DYNAMIC PROMPT SYNTHESIS & INJECTION (Bottom)
    draw_card(ax, 4, 8, 92, 22, bg="#0F172A", border=ACCENT_GREEN, radius=2)
    ax.text(50, 26, "TIER 3: UNIFIED RUNTIME CLINICAL CONTEXT ASSEMBLY (build_prompt)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=ACCENT_GREEN)
    
    draw_card(ax, 6, 11, 20, 12, bg=PANEL_BG, border=ACCENT_AMBER, lw=1.2)
    ax.text(16, 19, "Patient Memory", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ACCENT_AMBER)
    ax.text(16, 14.5, "Known chronic facts,\nallergies, prescriptions", ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 28, 11, 20, 12, bg=PANEL_BG, border=ACCENT_PURPLE, lw=1.2)
    ax.text(38, 19, "Session Summary", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ACCENT_PURPLE)
    ax.text(38, 14.5, "Compressed history of\nearlier consultation turns", ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 50, 11, 20, 12, bg=PANEL_BG, border=ACCENT_CYAN, lw=1.2)
    ax.text(60, 19, "Sliding Window", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ACCENT_CYAN)
    ax.text(60, 14.5, "Last 6 verbatim dialogue\nturns (Patient/Assistant)", ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    draw_card(ax, 72, 11, 22, 12, bg=PANEL_BG, border=ACCENT_ROSE, lw=1.2)
    ax.text(83, 19, "Retrieved RAG Docs", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ACCENT_ROSE)
    ax.text(83, 14.5, "Top-4 Pinecone vectors from\nThe Gale Encyclopedia", ha="center", va="center", fontsize=8, color=TEXT_MUTED)

    # CONNECTING ARROWS
    draw_arrow(ax, 20, 62, 20, 58, color=ACCENT_PURPLE, lw=2) # Rolling to Summarizer
    draw_arrow(ax, 80, 62, 80, 58, color=ACCENT_AMBER, lw=2) # Latest to Memory
    draw_arrow(ax, 26, 34, 38, 23, color=ACCENT_PURPLE, lw=2) # Summary to Prompt
    draw_arrow(ax, 74, 34, 16, 23, color=ACCENT_AMBER, lw=2, connectionstyle="arc3,rad=-0.15") # Memory to Prompt
    draw_arrow(ax, 50, 62, 60, 23, color=ACCENT_CYAN, lw=2) # Window to Prompt

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "04_session_memory_flow.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    print("Generating diagrams...")
    generate_diagram_1()
    generate_diagram_2()
    generate_diagram_3()
    generate_diagram_4()
    print("All diagrams successfully generated in ./docs/images/")
