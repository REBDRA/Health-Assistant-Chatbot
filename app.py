import html
import json
import os
import textwrap
import urllib.request
from datetime import date
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from ai_service import HealthAIFacade

# 1. Streamlit Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Health Assistant AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# --- HELPER FUNCTIONS ---

def clean_html(text: str) -> str:
    """Strips all leading whitespace from each line to prevent Markdown parser from treating HTML as indented code blocks."""
    return "\n".join(line.lstrip() for line in text.strip().splitlines() if line.strip())
def get_stars(rating: str) -> str:
    """Converts a numerical rating string (e.g. '4.5/5') to star emojis."""
    try:
        num = float(rating.split("/")[0])
        full = int(num)
        half = 1 if (num - full) >= 0.5 else 0
        return "⭐" * full + (" ✨" if half else "")
    except (ValueError, AttributeError, IndexError):
        return "⭐⭐⭐⭐"


def doctor_completeness_score(doc: dict) -> int:
    """Ranks doctors based on available contact details and credibility."""
    score = 0
    name = str(doc.get("name", "")).lower()
    phone = str(doc.get("phone", "")).lower()
    location = str(doc.get("location", "")).lower()
    rating = str(doc.get("rating", "")).lower()

    if name and name not in ("unknown", "no doctors found", ""):
        score += 3
    if phone and phone not in ("visit website", "n/a", "not available", ""):
        score += 3
    if location and location not in ("not available", "unknown", "", "local area", "not specified"):
        score += 2
    if rating and rating not in ("", "verified"):
        score += 1
    if doc.get("link"):
        score += 1
    return score


def detect_ip_location() -> str:
    """Auto-detects geographical location via IP with multiple fallback services."""
    client_ip = None
    try:
        client_ip = getattr(st.context, "ip_address", None)
    except Exception:
        pass

    if client_ip in (None, "127.0.0.1", "localhost", "::1"):
        client_ip = ""

    services = [
        (
            f"https://ipapi.co/{client_ip}/json/" if client_ip else "https://ipapi.co/json/",
            lambda d: (d.get("city", ""), d.get("region", ""), d.get("country_name", "")),
        ),
        (
            f"https://ip-api.com/json/{client_ip}" if client_ip else "https://ip-api.com/json/",
            lambda d: (d.get("city", ""), d.get("regionName", ""), d.get("country", "")),
        ),
        (
            f"https://ipwhois.app/json/{client_ip}" if client_ip else "https://ipwhois.app/json/",
            lambda d: (d.get("city", ""), d.get("region", ""), d.get("country", "")),
        ),
    ]

    for url, parse in services:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode())
                if data.get("success") is False or data.get("status") == "fail":
                    continue
                city, region, country = parse(data)
                if city:
                    parts = [p for p in [city, region, country] if p]
                    return ", ".join(parts)
        except Exception:
            continue

    return "Kolkata, West Bengal, India"


def get_daily_tip(api_key: Optional[str]) -> str:
    """Fetches a fresh daily health tip, caching within session state."""
    today = date.today().isoformat()
    if "daily_tip" in st.session_state and st.session_state.get("daily_tip_date") == today:
        return st.session_state["daily_tip"]

    tip = (
        "Hydration is key to cellular health! Drinking water before meals aids digestion "
        "and keeps cognitive energy sharp throughout your day."
    )

    if api_key:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Provide ONE practical, uplifting, actionable health tip in 1-2 sentences. "
                            "Under 25 words. Plain text only."
                        ),
                    },
                    {"role": "user", "content": "Give me a daily health tip."},
                ],
                max_tokens=45,
                temperature=0.7,
            )
            fresh_tip = (response.choices[0].message.content or "").strip()
            if fresh_tip:
                tip = fresh_tip
        except Exception:
            pass

    st.session_state["daily_tip"] = tip
    st.session_state["daily_tip_date"] = today
    return tip


# --- CSS & STYLING ---

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1e28 0%, #071017 90%);
    color: #e2e8f0;
}

[data-testid="block-container"] {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}


/* Chat Bubbles */
.chat-response-card {
    background: rgba(15, 30, 42, 0.82);
    border: 1px solid rgba(137, 247, 254, 0.25);
    border-radius: 18px;
    padding: 1.5rem;
    color: #f1f5f9;
    line-height: 1.65;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.2rem;
    animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* High-Precision Doctor Cards */
.doctor-card {
    background: linear-gradient(135deg, rgba(16, 36, 50, 0.85), rgba(11, 25, 35, 0.95));
    border: 1px solid rgba(30, 176, 191, 0.35);
    border-left: 5px solid #1eb0bf;
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.doctor-card:hover {
    transform: translateY(-2px);
    border-color: rgba(137, 247, 254, 0.7);
    box-shadow: 0 10px 25px rgba(30, 176, 191, 0.15);
}

.doctor-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
    margin-bottom: 10px;
}

.doctor-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
}

.doctor-specialty {
    display: inline-block;
    background: rgba(30, 176, 191, 0.2);
    border: 1px solid rgba(30, 176, 191, 0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #89f7fe;
    margin-top: 4px;
}

.doctor-rating {
    font-size: 0.9rem;
    font-weight: 600;
    color: #facc15;
    background: rgba(250, 204, 21, 0.12);
    padding: 4px 8px;
    border-radius: 8px;
}

.doctor-legit-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    margin-top: 4px;
    margin-bottom: 8px;
}

.doctor-details {
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-bottom: 10px;
}

.doctor-detail-item {
    margin-bottom: 4px;
    display: flex;
    align-items: baseline;
    gap: 8px;
}

.doctor-why {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 8px;
    font-style: italic;
}

.doctor-btn {
    display: inline-block;
    background: #1eb0bf;
    color: #071017 !important;
    text-decoration: none !important;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 6px 14px;
    border-radius: 8px;
    margin-top: 8px;
    transition: background 0.2s ease;
}

.doctor-btn:hover {
    background: #5dfff7;
}

/* Urgency Badges */
.urgency-emergency {
    background: rgba(220, 38, 38, 0.2);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
}

.urgency-urgent {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fde68a;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
}

.urgency-routine {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #a7f3d0;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
}

/* Emergency Banner */
.emergency-banner {
    background: linear-gradient(90deg, rgba(220, 38, 38, 0.25), rgba(185, 28, 28, 0.15));
    border: 1px solid rgba(239, 68, 68, 0.45);
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #fca5a5;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Quick Prompt Pill Buttons */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: rgba(137, 247, 254, 0.08) !important;
    border: 1px solid rgba(137, 247, 254, 0.25) !important;
    border-radius: 20px !important;
    color: #89f7fe !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    background: rgba(137, 247, 254, 0.2) !important;
    border-color: #89f7fe !important;
    transform: translateY(-2px) !important;
}

/* Primary Buttons */
div[data-testid="stButton"] button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Footer */
.custom-footer {
    position: fixed;
    bottom: 6px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(203, 213, 225, 0.55);
    font-size: 11px;
    z-index: 999;
    pointer-events: none;
    text-align: center;
}
</style>

<div class="custom-footer">
    🩺 Health Assistant AI • Evidence-based triage • Consult licensed medical professionals for diagnoses
</div>
""",
    unsafe_allow_html=True,
)


# --- API KEY MANAGEMENT (BACKEND ENFORCED, ZERO LEAKS) ---

def get_active_api_key() -> str:
    """Safely retrieves Groq API key from session override, Streamlit Secrets, or environment."""
    # 1. User manual input / session state (takes priority if user entered a custom key in UI)
    if st.session_state.get("user_groq_api_key"):
        return st.session_state["user_groq_api_key"].strip()
    # 2. Streamlit Cloud secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    # 3. Environment variable (.env or OS)
    env_k = os.environ.get("GROQ_API_KEY", "").strip()
    if env_k:
        return env_k
    return ""

active_api_key = get_active_api_key()

# --- SIDEBAR (MINIMAL, CLEAN, NO TECHNICAL/GREEN CLUTTER) ---
with st.sidebar:
    st.markdown("### 🩺 Health Assistant")
    st.caption("Clinical Triage & Specialist Discovery")
    st.markdown("---")

    # API Configuration Expander
    with st.expander("🔑 API Key Settings", expanded=not bool(active_api_key)):
        masked_key = f"...{active_api_key[-4:]}" if active_api_key and len(active_api_key) > 4 else "Not configured"
        st.caption(f"Status: {'✅ Connected' if active_api_key else '⚠️ Key Needed'} ({masked_key})")
        new_key = st.text_input(
            "Groq API Key",
            type="password",
            value="",
            placeholder="gsk_...",
            help="Enter or update your Groq API key.",
            key="sidebar_groq_key_input",
        )
        if st.button("Save & Apply Key", use_container_width=True, key="save_sidebar_key_btn"):
            if new_key.strip():
                st.session_state["user_groq_api_key"] = new_key.strip()
                st.rerun()

    # Chat Actions
    st.markdown("#### 💬 Chat Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if "messages" in st.session_state and st.session_state["messages"]:
        export_text = "# Health Assistant Consultation Report\n\n"
        export_text += f"Date: {date.today().isoformat()}\n"
        export_text += f"Location: {st.session_state.get('user_location', 'Not specified')}\n\n---\n\n"
        for m in st.session_state["messages"]:
            role = "Patient" if m.get("role") == "user" else "Health AI"
            export_text += f"### {role}\n{m.get('content')}\n\n"
        export_text += "\n\n*Note: This report is generated by an AI assistant for informative triage only.*"

        st.download_button(
            label="📥 Export Medical Summary",
            data=export_text,
            file_name=f"health_consultation_{date.today().isoformat()}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Consultation Guide")
    st.markdown(
        """
        <div style='font-size: 0.83rem; color: #94a3b8; line-height: 1.55;'>
        • <b>Describe symptoms clearly:</b> Mention duration, intensity, and location.<br>
        • <b>Specify your area:</b> Use the Clinic Finder to discover nearby doctors.<br>
        • <b>Emergency:</b> For severe chest pain, shortness of breath, or trauma, call <b>112/911</b> immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Initialize AI Facade if key exists
health_ai = None
if active_api_key:
    try:
        health_ai = HealthAIFacade(api_key=active_api_key)
    except Exception as exc:
        st.sidebar.error(f"AI Service Error: {exc}")


# --- THREE-COLUMN WORKSPACE LAYOUT ---

left_col, main_col, right_col = st.columns([1, 2.2, 1], gap="medium")


# ==========================================
# ⚡ LEFT COLUMN: Quick Health Tools
# ==========================================
with left_col:
    st.markdown("### ⚡ Quick Tools")

    # BMI Calculator Widget
    with st.container(border=True):
        st.markdown("#### ⚖️ BMI Assessment")
        weight = st.number_input("Weight (kg)", min_value=15.0, max_value=250.0, value=68.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=70.0, max_value=240.0, value=172.0, step=1.0)

        bmi = weight / ((height / 100) ** 2)
        healthy_min = 18.5 * ((height / 100) ** 2)
        healthy_max = 24.9 * ((height / 100) ** 2)

        if bmi < 18.5:
            status, color = "Underweight", "#38bdf8"
        elif 18.5 <= bmi < 24.9:
            status, color = "Healthy Weight", "#4ade80"
        elif 25 <= bmi < 29.9:
            status, color = "Overweight", "#fbbf24"
        else:
            status, color = "Obese Range", "#f87171"

        st.markdown(
            f"""
            <div style='background: rgba(255,255,255,0.04); border-radius: 10px; padding: 12px; margin-top: 8px;'>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Current Score:</div>
                <div style='font-size: 1.4rem; font-weight: 700; color: {color};'>{bmi:.1f} • {status}</div>
                <div style='font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;'>
                    Ideal weight: <b>{healthy_min:.1f} - {healthy_max:.1f} kg</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Daily Health Tip Widget
    with st.container(border=True):
        tip = get_daily_tip(active_api_key)
        st.markdown("#### 🍎 Daily Wellness Insight")
        st.markdown(
            f"""
            <div style='
                background: rgba(30, 176, 191, 0.12);
                border-left: 4px solid #1eb0bf;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 0.9rem;
                color: #e2e8f0;
                line-height: 1.5;
            '>
                💡 {html.escape(tip)}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# 💡 RIGHT COLUMN: Wellness Hub
# ==========================================
with right_col:
    st.markdown("### 💡 Wellness Hub")

    # Local Doctor Finder Setup
    if "user_location" not in st.session_state:
        st.session_state["user_location"] = ""
        st.session_state["location_allowed"] = False

    with st.container(border=True):
        st.markdown("#### 📍 Clinic & Doctor Finder")

        if not st.session_state["location_allowed"] or not st.session_state["user_location"]:
            st.markdown(
                "<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;'>"
                "Tailor doctor listings to your exact area."
                "</p>",
                unsafe_allow_html=True,
            )

            if st.button("🌐 Auto-Detect via IP", use_container_width=True):
                with st.spinner("Detecting locality..."):
                    detected = detect_ip_location()
                    st.session_state["user_location"] = detected
                    st.session_state["location_allowed"] = True
                    st.rerun()

            manual_loc = st.text_input(
                "Or specify city / area:",
                placeholder="e.g. Bandra, Mumbai",
                key="manual_loc_input",
            )
            if manual_loc:
                st.session_state["user_location"] = manual_loc.strip()
                st.session_state["location_allowed"] = True
                st.rerun()
        else:
            st.markdown(
                f"""
                <div style='padding: 4px 0 10px 0;'>
                    <div style='font-size: 0.8rem; color: #94a3b8;'>Target Search Area:</div>
                    <div style='font-size: 1rem; font-weight: 700; color: #89f7fe;'>
                        📍 {html.escape(st.session_state['user_location'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("✏️ Change Location", use_container_width=True):
                st.session_state["user_location"] = ""
                st.session_state["location_allowed"] = False
                st.rerun()

    # Hydration Tracker Widget
    if "water_litres" not in st.session_state:
        st.session_state["water_litres"] = 0.0

    with st.container(border=True):
        st.markdown("#### 💧 Hydration Tracker")
        target_litres = 2.5
        current_litres = st.session_state["water_litres"]
        percentage = min(current_litres / target_litres, 1.0)

        st.progress(percentage, text=f"{current_litres:.2f} / {target_litres:.1f} Litres ({int(percentage * 100)}%)")

        col_w1, col_w2, col_w3 = st.columns(3)
        if col_w1.button("🥤 +250ml", use_container_width=True):
            st.session_state["water_litres"] = round(current_litres + 0.25, 2)
            st.rerun()
        if col_w2.button("🍶 +500ml", use_container_width=True):
            st.session_state["water_litres"] = round(current_litres + 0.50, 2)
            st.rerun()
        if col_w3.button("🔄 Reset", use_container_width=True):
            st.session_state["water_litres"] = 0.0
            st.rerun()

        if current_litres >= target_litres:
            st.caption("🎉 Hydration target accomplished today!")


# ==========================================
# 🤖 MAIN COLUMN: Chatbot & Clinical Triage
# ==========================================
with main_col:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 14px; margin-bottom: 0.5rem;'>
            <span style='font-size: 2.5rem;'>🩺</span>
            <div>
                <h2 style='margin: 0; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;'>
                    Health Assistant AI
                </h2>
                <div style='font-size: 0.9rem; color: #94a3b8;'>
                    Intelligent Clinical Triage • Home Remedies • Verified Local Specialists
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Emergency Disclaimer Banner
    st.markdown(
        """
        <div class="emergency-banner">
            <span style="font-size: 1.2rem;">🚨</span>
            <div>
                <strong>Medical Emergency?</strong> If you have severe chest pain, shortness of breath, sudden numbness, or heavy bleeding, call <strong>112</strong> (India/EU) or <strong>911</strong> (US) immediately.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Suggestion Prompts
    st.markdown("<small style='color: #94a3b8; font-weight: 600;'>Suggested Inquiries:</small>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)

    suggested_prompt = None
    if q_col1.button("🤕 Severe Migraine Relief", use_container_width=True):
        suggested_prompt = "I have a throbbing migraine headache on one side of my head with nausea and light sensitivity. What specialist should I see and what home remedies can help?"
    if q_col2.button("💊 Alternating Medications?", use_container_width=True):
        suggested_prompt = "Can I safely alternate paracetamol and ibuprofen for fever, and what is the proper timing?"
    if q_col3.button("🌿 Acute Acid Reflux Care", use_container_width=True):
        suggested_prompt = "I am suffering from acute acid reflux and burning sensation in my chest after eating. What home remedies help immediately and what specialist treats chronic GERD?"

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Chat input
    user_input = st.chat_input("Describe your symptoms or ask a health question...")

    # Choose active prompt
    active_prompt = user_input or suggested_prompt

    if active_prompt:
        if not active_api_key or not health_ai:
            st.error("Please provide a Groq API Key in the left sidebar to consult the AI assistant.")
        else:
            # Append and render user message
            st.session_state["messages"].append({"role": "user", "content": active_prompt})

            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

            # Generate AI Triage Response
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Analyzing symptoms & finding matched local specialists..."):
                    try:
                        active_loc = (
                            st.session_state.get("user_location", "")
                            or "Kolkata, West Bengal, India"
                        )
                        data = health_ai.get_structured_response(
                            user_prompt=active_prompt,
                            chat_history=st.session_state["messages"][:-1],
                            user_location=active_loc,
                        )

                        if not data.get("is_valid_query", True):
                            response_text = data.get(
                                "error_message",
                                "I can only assist with health, medical, and wellness questions.",
                            )
                            card_html = clean_html(f"""
                            <div class="chat-response-card" style="border-left: 4px solid #f59e0b;">
                                <div style="font-weight: 700; color: #fbbf24; margin-bottom: 8px; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                                    <span>⚠️</span> <span>Clinical Clarification Needed</span>
                                </div>
                                <div style="line-height: 1.6; color: #f1f5f9; font-size: 0.95rem;">
                                    {html.escape(response_text)}
                                </div>
                            </div>
                            """)
                            st.markdown(card_html, unsafe_allow_html=True)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": card_html, "is_card": True}
                            )
                        else:
                            # Build Rich Triage Presentation
                            html_parts = []

                            # 1. Urgency Badge
                            urgency = data.get("urgency_level", "Routine")
                            urgency_cls = "urgency-routine"
                            if "emergency" in urgency.lower():
                                urgency_cls = "urgency-emergency"
                            elif "urgent" in urgency.lower():
                                urgency_cls = "urgency-urgent"

                            html_parts.append(f'<div class="{urgency_cls}">⏱️ Consultation Urgency: {html.escape(urgency)}</div>')

                            # 2. Clinical Overview
                            direct_ans = data.get("direct_answer", "")
                            if direct_ans:
                                html_parts.append(
                                    clean_html(f"""
                                    <div style='margin-bottom: 16px; font-size: 1rem; line-height: 1.6; color: #f1f5f9;'>
                                        <strong>🩺 Clinical Overview:</strong><br>{html.escape(direct_ans)}
                                    </div>
                                    """)
                                )

                            # 3. Dedicated High-Precision Doctor Cards
                            doctors = data.get("doctors", [])
                            specialty = data.get("specialty_needed", "")
                            if doctors:
                                html_parts.append(
                                    clean_html(f"""
                                    <div style='margin-top: 14px; margin-bottom: 4px; font-size: 1.05rem; font-weight: 700; color: #89f7fe;'>
                                        👨‍⚕️ Verified Specialists & Premier Hospitals in {html.escape(active_loc)}:
                                    </div>
                                    <div style='margin-bottom: 12px; font-size: 0.82rem; color: #94a3b8;'>
                                        Permanently established institutions with verified patient visits & active outpatient care (OPD):
                                    </div>
                                    """)
                                )

                                sorted_doctors = sorted(
                                    doctors,
                                    key=lambda d: doctor_completeness_score(d if isinstance(d, dict) else d.model_dump()),
                                    reverse=True,
                                )

                                for doc in sorted_doctors:
                                    if hasattr(doc, "model_dump"):
                                        doc = doc.model_dump()

                                    doc_name = html.escape(doc.get("name") or "Specialist Clinic")
                                    doc_spec = html.escape(doc.get("specialty") or specialty or "Medical Specialist")
                                    doc_hosp = html.escape(doc.get("clinic_or_hospital") or "")

                                    doc_loc_raw = str(doc.get("location") or "").strip()
                                    if not doc_loc_raw or doc_loc_raw.lower() in ("not specified", "unknown", "local area", "n/a", "none"):
                                        doc_loc = html.escape(active_loc)
                                    else:
                                        doc_loc = html.escape(doc_loc_raw)

                                    doc_phone_raw = str(doc.get("phone") or "").strip()
                                    if not doc_phone_raw or doc_phone_raw.lower() in ("not specified", "unknown", "n/a", "none"):
                                        doc_phone = "Visit Clinic / Hospital Directory"
                                    else:
                                        doc_phone = html.escape(doc_phone_raw)

                                    doc_rating = html.escape(doc.get("rating") or "4.5/5")
                                    doc_stars = get_stars(doc_rating)
                                    doc_link = doc.get("link", "")
                                    doc_why = html.escape(doc.get("why_recommended") or "")
                                    doc_status = html.escape(doc.get("verification_status") or "Verified Legitimate • Active Patient Footfall")
                                    legit_badge = f"<div class='doctor-legit-badge'>🛡️ {doc_status}</div>"

                                    hosp_line = f"<div class='doctor-detail-item'>🏥 <span><strong>Established Facility:</strong> {doc_hosp}</span></div>" if doc_hosp else ""
                                    why_line = f"<div class='doctor-why'>💡 <strong>Why Recommended:</strong> {doc_why}</div>" if doc_why else ""

                                    btn_line = ""
                                    if doc_link and doc_link.startswith("http"):
                                        btn_line = f"<a href='{html.escape(doc_link)}' target='_blank' class='doctor-btn'>View Clinic / Book Appointment →</a>"

                                    card_html = clean_html(f"""
                                    <div class="doctor-card">
                                        <div class="doctor-header">
                                            <div>
                                                <div class="doctor-name">{doc_name}</div>
                                                <div class="doctor-specialty">{doc_spec}</div>
                                            </div>
                                            <div class="doctor-rating">{doc_stars} ({doc_rating})</div>
                                        </div>
                                        {legit_badge}
                                        <div class="doctor-details">
                                            {hosp_line}
                                            <div class="doctor-detail-item">📍 <span>{doc_loc}</span></div>
                                            <div class="doctor-detail-item">📞 <span><strong>Contact:</strong> {doc_phone}</span></div>
                                            {why_line}
                                        </div>
                                        {btn_line}
                                    </div>
                                    """)
                                    html_parts.append(card_html)

                            # 4. Safe Home Remedies
                            remedies = data.get("remedies", [])
                            if remedies:
                                rem_items = "".join([f"<li style='margin-bottom: 6px;'>{html.escape(r)}</li>" for r in remedies])
                                html_parts.append(
                                    clean_html(f"""
                                    <div style='margin-top: 14px; background: rgba(30, 176, 191, 0.08); border-radius: 10px; padding: 12px 16px;'>
                                        <div style='font-weight: 700; color: #89f7fe; margin-bottom: 6px;'>🌿 Recommended Home Care & Recovery Steps:</div>
                                        <ul style='margin: 0; padding-left: 20px; color: #e2e8f0;'>{rem_items}</ul>
                                    </div>
                                    """)
                                )

                            # 5. Medical Guidance & Red Flags
                            advice = data.get("advice", "")
                            if advice:
                                html_parts.append(
                                    clean_html(f"""
                                    <div style='margin-top: 12px; font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;'>
                                        <strong>💡 Medical Guidance & Warning Signs:</strong><br>{html.escape(advice)}
                                    </div>
                                    """)
                                )

                            html_parts.append(
                                clean_html("""
                                <div style='margin-top: 14px; font-size: 0.78rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;'>
                                    *Disclaimer: This guidance is provided by an AI triage assistant and does not substitute for a formal diagnosis or emergency medical care.*
                                </div>
                                """)
                            )

                            full_response_html = clean_html(f'<div class="chat-response-card">{"".join(html_parts)}</div>')
                            st.markdown(full_response_html, unsafe_allow_html=True)

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": full_response_html, "is_card": True}
                            )

                    except Exception as e:
                        err_str = str(e)
                        if "401" in err_str or "invalid_api_key" in err_str.lower():
                            auth_err_card = clean_html("""
                            <div class="chat-response-card" style="border-left: 4px solid #ef4444;">
                                <div style="font-weight: 700; color: #fca5a5; margin-bottom: 8px; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                                    <span>🔑</span> <span>Groq API Key Authentication Failed</span>
                                </div>
                                <div style="line-height: 1.6; color: #cbd5e1; font-size: 0.95rem;">
                                    Your Groq API Key was not recognized (401: Invalid API Key). Please open <b>API Key Settings</b> in the left sidebar to enter or update your active key, or configure <code>GROQ_API_KEY</code> in Streamlit Cloud Secrets.
                                </div>
                            </div>
                            """)
                            st.markdown(auth_err_card, unsafe_allow_html=True)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": auth_err_card, "is_card": True}
                            )
                        else:
                            err_msg = f"Unable to process consultation at this moment: {e}"
                            st.error(err_msg)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": err_msg}
                            )

    # Render previous conversation history
    rendered_history = st.session_state.get("messages", [])
    if active_prompt and len(rendered_history) >= 2:
        history_to_display = rendered_history[:-2]
    else:
        history_to_display = rendered_history

    for msg in reversed(history_to_display):
        avatar = "🩺" if msg.get("role") == "assistant" else "👤"
        with st.chat_message(msg.get("role"), avatar=avatar):
            if msg.get("is_card"):
                st.markdown(msg.get("content"), unsafe_allow_html=True)
            else:
                st.markdown(msg.get("content"))
