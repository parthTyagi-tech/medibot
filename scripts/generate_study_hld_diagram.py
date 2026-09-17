"""
Generate high-resolution architecture diagram for MediAssist Study Notes & HLD.
Outputs docs/images/05_study_notes_and_hld_architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "05_study_notes_and_hld_architecture.png")

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

def generate_hld_diagram():
    fig, ax = plt.subplots(figsize=(18, 14), dpi=220)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_card(x, y, w, h, bg=PANEL_BG, border=BORDER_COLOR, lw=1.5, radius=1.2, alpha=1.0):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, alpha=alpha, zorder=2
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color=ACCENT_CYAN, lw=1.8, style="->", connectionstyle="arc3,rad=0"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style, color=color, lw=lw,
                connectionstyle=connectionstyle,
                shrinkA=3, shrinkB=3
            ),
            zorder=4
        )

    # Title Banner
    draw_card(4, 91.5, 92, 6.5, bg="#1E293B", border=ACCENT_BLUE, lw=2.0)
    ax.text(50, 95.8, "MediAssist: End-to-End System Architecture & Fault-Tolerant HLD", 
            ha="center", va="center", fontsize=15, fontweight="bold", color=TEXT_WHITE)
    ax.text(50, 93.0, "Dual-Swimlane Mapping: Real-Time Flow & Failover Logic (Left) vs. Exact Codebase Module Implementation (Right)", 
            ha="center", va="center", fontsize=10, color=ACCENT_BLUE)

    # Column Headers
    draw_card(4, 85, 44, 4.5, bg="#0F172A", border=ACCENT_CYAN, lw=1.5)
    ax.text(26, 87.2, "SYSTEM HIGH-LEVEL DESIGN (HLD) & FAILOVER FLOW", 
            ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_CYAN)

    draw_card(52, 85, 44, 4.5, bg="#0F172A", border=ACCENT_PURPLE, lw=1.5)
    ax.text(74, 87.2, "EXACT CODEBASE MODULE & REPOSITORY MAPPING", 
            ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT_PURPLE)

    # Steps Definitions
    steps = [
        {
            "num": "STEP 1",
            "title": "Network Ingress & Proxy Sanitization",
            "desc": "Browser POST /get -> Reverse Proxy (Nginx/Render)\nProxyFix extracts client IP, sets HTTPS scheme, stops OAuth loop",
            "file": "app.py",
            "details": "ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)\nValidates client IP & preserves secure OAuth redirect URIs",
            "color": ACCENT_BLUE,
            "y": 75
        },
        {
            "num": "STEP 2",
            "title": "Deterministic Input Guardrails",
            "desc": "Pre-compiled regex scans in 0.001ms:\n1. Prompt Injection / DAN mode  2. Illegal Drug / Poison\n3. Acute 911 Emergency (Stroke FAST, Chest Pain, Bleeding)",
            "file": "research/src/guardrails.py",
            "details": "apply_input_guardrails(user_input)\nCOMPILED_INJECTION_PATTERNS & detect_medical_emergency()",
            "color": ACCENT_RED,
            "y": 64.5
        },
        {
            "num": "STEP 3",
            "title": "Two-Tier Hybrid Intent Router",
            "desc": "Tier 1: 0ms Regex Token Matcher (greetings, non-medical)\nTier 2: Groq Compound Mini LLM fallback (<80ms)\nSaves 95% latency & token cost on casual chat",
            "file": "research/src/intent_classifier.py",
            "details": "classify_intent(classifierModel, msg)\nReturns: 'greeting' | 'general_query' | 'medical_query'",
            "color": ACCENT_AMBER,
            "y": 54
        },
        {
            "num": "STEP 4",
            "title": "Clinical Triage & Patient State Engine",
            "desc": "Persists PatientState (age, pregnancy, chemo) across turns\nTriggers: Febrile Neutropenia, Neonatal Fever (<3mo), Preeclampsia\nBlocks pediatric exact mg dosing (strictly weight-based mg/kg)",
            "file": "research/src/clinical_triage.py",
            "details": "PatientState, AUDITABLE_TRIAGE_MATRIX\nextract_patient_state(), evaluate_triage_tier()\ncheck_medication_contraindications()",
            "color": ACCENT_GREEN,
            "y": 43.5
        },
        {
            "num": "STEP 5",
            "title": "Vector RAG Search & Offline Fallback",
            "desc": "Text -> 384D unit vector (LocalEmbeddings)\nQuery Pinecone 'medical-chatbot' (top_k=4)\nFAILOVER: If Pinecone is down, synthesizes clinical guide on the fly!",
            "file": "research/src/helper.py & services/ai_service.py",
            "details": "LocalEmbeddings.embed_query() (helper.py)\nCustomPineconeRetriever._get_relevant_documents()\ntry/except fallback to Gale Encyclopedia principles",
            "color": ACCENT_CYAN,
            "y": 33
        },
        {
            "num": "STEP 6",
            "title": "Dynamic Context Assembly (Prompt Builder)",
            "desc": "Escapes user braces { -> {{ to prevent template crashes\nInjects patient state, triage risk directive (ER at top if Urgent)\nEnforces: Triage first, under 120 words, NO definitive diagnosis",
            "file": "services/ai_service.py",
            "details": "build_prompt(history, memory, user, patient_state)\nChatPromptTemplate with strict clinical doctor system prompt",
            "color": ACCENT_PURPLE,
            "y": 22.5
        },
        {
            "num": "STEP 7",
            "title": "Resilient LLM Inference with Failover",
            "desc": "Primary Model: Groq LPU (openai/gpt-oss-20b) for <300ms inference\nFAILOVER: If 429 quota error, auto-switches to gpt-oss-120b!\nZero rate-limit crashes for the patient",
            "file": "services/ai_service.py",
            "details": "class GroqChatModel(BaseChatModel)\n_generate() with try/except automatic fallback invocation",
            "color": ACCENT_BLUE,
            "y": 12
        },
        {
            "num": "STEP 8",
            "title": "Output Sanitization & Client Delivery",
            "desc": "1. Strips leaked prompt tags (<system_prompt>, <think>)\n2. Rewrites 'You have X' -> 'Symptoms commonly associated with X'\n3. Appends legal medical disclaimer -> Displays safely in UI!",
            "file": "research/src/guardrails.py & routes/chat.py",
            "details": "apply_output_guardrails(raw_answer, is_medical=True)\nroutes/chat.py returns sanitized clinical response to browser",
            "color": ACCENT_GREEN,
            "y": 1.5
        }
    ]

    for i, s in enumerate(steps):
        y = s["y"]
        
        # Left card (Flow)
        draw_card(4, y, 44, 9, bg=PANEL_BG, border=s["color"], lw=1.4)
        ax.text(6, y + 7.2, f"{s['num']}: {s['title']}", fontsize=10.5, fontweight="bold", color=s["color"])
        ax.text(6, y + 3.8, s["desc"], fontsize=8.5, color=TEXT_WHITE, va="center", linespacing=1.35)

        # Right card (Codebase Mapping)
        draw_card(52, y, 44, 9, bg=PANEL_BG, border=BORDER_COLOR, lw=1.2)
        ax.text(54, y + 7.2, f"Module: {s['file']}", fontsize=10.5, fontweight="bold", color=ACCENT_PURPLE)
        ax.text(54, y + 3.8, s["details"], fontsize=8.5, color=TEXT_MUTED, va="center", linespacing=1.35)

        # Horizontal connecting arrow (Flow -> Code)
        draw_arrow(48, y + 4.5, 52, y + 4.5, color=s["color"], lw=1.6)

        # Vertical flow arrow to next step
        if i < len(steps) - 1:
            next_y = steps[i+1]["y"]
            draw_arrow(26, y, 26, next_y + 9, color=ACCENT_CYAN, lw=2.0)

    plt.tight_layout()
    plt.savefig(OUT_PATH, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Success! HLD Diagram generated: {OUT_PATH}")

if __name__ == "__main__":
    generate_hld_diagram()
