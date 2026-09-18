import html
import json
import os
import urllib.request
from datetime import date
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import ai_service
from ai_service import HealthAIFacade

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="Health Assistant AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# --- HELPER FUNCTIONS ---

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
    name = doc.get("name", "").lower()
    phone = doc.get("phone", "").lower()
    location = doc.get("location", "").lower()
    rating = doc.get("rating", "").lower()

    if name and name not in ("unknown", "no doctors found", ""):
        score += 3
    if phone and phone not in ("visit website", "n/a", "not available", ""):
        score += 3
    if location and location not in ("not available", "unknown", "", "not specified"):
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
            tip_model = "openai/gpt-oss-20b"
            response = client.chat.completions.create(
                model=tip_model,
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
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(19, 35, 46, 0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(137, 247, 254, 0.22);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(137, 247, 254, 0.5);
}

/* Chat Bubbles */
.chat-response-card {
    background: rgba(15, 30, 42, 0.82);
    border: 1px solid rgba(137, 247, 254, 0.28);
    border-radius: 18px;
    padding: 1.4rem;
    color: #f1f5f9;
    line-height: 1.65;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
    animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
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


# --- API KEY MANAGEMENT ---

env_key = os.environ.get("GROQ_API_KEY")
try:
    secret_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    secret_key = None

stored_key = env_key or secret_key or st.session_state.get("user_groq_api_key", "")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Settings")

    user_input_key = st.text_input(
        "Groq API Key",
        value=stored_key,
        type="password",
        help="Get your free key from https://console.groq.com/keys",
        placeholder="gsk_...",
    )

    if user_input_key != stored_key:
        st.session_state["user_groq_api_key"] = user_input_key.strip()
        st.rerun()

    active_api_key = st.session_state.get("user_groq_api_key") or stored_key

    if active_api_key:
        st.success("🟢 API Key Active", icon="✅")
    else:
        st.warning("🟠 API Key Missing", icon="⚠️")
        st.markdown(
            """
            <small style='color: #94a3b8;'>
            To enable AI diagnoses, enter a free Groq API key above or set <code>GROQ_API_KEY</code> in <code>.env</code>.
            <br><a href='https://console.groq.com/keys' target='_blank' style='color: #89f7fe;'>Get Free Groq Key →</a>
            </small>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Clear Chat & Consultation Export
    st.markdown("### 💬 Chat Management")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if "messages" in st.session_state and st.session_state["messages"]:
        # Prepare consultation export text
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


# Initialize AI Facade if key exists
health_ai = None
if active_api_key:
    try:
        health_ai = HealthAIFacade(api_key=active_api_key)
    except Exception as exc:
        st.sidebar.error(f"AI Service Initialization Error: {exc}")


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
            status, color, alert_type = "Underweight", "#38bdf8", "info"
        elif 18.5 <= bmi < 24.9:
            status, color, alert_type = "Healthy Weight", "#4ade80", "success"
        elif 25 <= bmi < 29.9:
            status, color, alert_type = "Overweight", "#fbbf24", "warning"
        else:
            status, color = "Obese Range", "#f87171"
            alert_type = "error"

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

    # API Key warning if missing
    if not active_api_key:
        st.info(
            "👋 **Welcome!** Please enter a free Groq API Key in the left sidebar to activate AI medical triage.",
            icon="🔑",
        )

    # Quick Suggestion Prompts
    st.markdown("<small style='color: #94a3b8; font-weight: 600;'>Suggested Inquiries:</small>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)

    suggested_prompt = None
    if q_col1.button("🤕 Severe Migraine Relief", use_container_width=True):
        suggested_prompt = "I have a throbbing migraine headache with sensitivity to light. What remedies help and should I see a neurologist?"
    if q_col2.button("💊 Paracetamol with Ibuprofen?", use_container_width=True):
        suggested_prompt = "Can I safely alternate paracetamol and ibuprofen for fever, and what is the proper timing?"
    if q_col3.button("🌿 Acidity & Acid Reflux Care", use_container_width=True):
        suggested_prompt = "I am suffering from acute acid reflux and burning sensation in my chest after eating. What home remedies help immediately?"

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Chat input
    user_input = st.chat_input("Describe your symptoms or ask a health question...")

    # Choose active prompt
    active_prompt = user_input or suggested_prompt

    if active_prompt:
        if not active_api_key or not health_ai:
            st.error("Please provide a valid Groq API Key in the left sidebar to consult the AI assistant.")
        else:
            # Append and render user message
            st.session_state["messages"].append({"role": "user", "content": active_prompt})

            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

            # Generate AI Triage Response
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Analyzing symptoms & scanning local specialists..."):
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
                            st.markdown(response_text)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response_text}
                            )
                        else:
                            content_blocks = []

                            # Direct Answer
                            direct_ans = data.get("direct_answer", "")
                            if direct_ans:
                                content_blocks.append(f"### 🩺 Clinical Overview\n{direct_ans}")

                            # Home Remedies
                            remedies = data.get("remedies", [])
                            if remedies:
                                rem_list = "\n".join([f"- **Step {i}:** {r}" for i, r in enumerate(remedies, 1)])
                                content_blocks.append(f"### 🌿 Recommended Actions & Remedies\n{rem_list}")

                            # Preventive Advice
                            advice = data.get("advice", "")
                            if advice:
                                content_blocks.append(f"### 💡 Medical Guidance & Red Flags\n{advice}")

                            # Local Doctors
                            doctors = data.get("doctors", [])
                            if doctors:
                                sorted_doctors = sorted(
                                    doctors,
                                    key=lambda d: doctor_completeness_score(d if isinstance(d, dict) else d.model_dump()),
                                    reverse=True,
                                )
                                doc_text = f"### 👨‍⚕️ Verified Specialists Near {active_loc}\n\n"
                                for doc in sorted_doctors:
                                    if hasattr(doc, "model_dump"):
                                        doc = doc.model_dump()
                                    stars = get_stars(doc.get("rating", ""))
                                    phone_val = doc.get("phone", "N/A")
                                    link_val = doc.get("link", "")

                                    if link_val:
                                        phone_display = f"{phone_val} • [Visit Profile / Booking]({link_val})"
                                    else:
                                        phone_display = phone_val

                                    doc_text += (
                                        f"**🧑‍⚕️ {doc.get('name', 'Specialist')}**\n\n"
                                        f"- 📍 **Address/Area:** {doc.get('location', 'Area nearby')}\n"
                                        f"- 📞 **Contact:** {phone_display}\n"
                                        f"- ⭐ **Rating:** {stars}\n\n---\n"
                                    )
                                content_blocks.append(doc_text)

                            content_blocks.append(
                                "\n*Disclaimer: This guidance is provided by an AI triage assistant and does not substitute for a formal diagnosis or emergency medical care.*"
                            )

                            full_response = "\n\n".join(content_blocks)
                            st.markdown(
                                f'<div class="chat-response-card">{full_response}</div>',
                                unsafe_allow_html=True,
                            )

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": full_response, "is_card": True}
                            )

                    except Exception as e:
                        err_msg = f"Unable to process consultation at this moment: {e}"
                        st.error(err_msg)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": err_msg}
                        )

    # Render previous conversation history
    # If a prompt was just submitted, exclude the last turn since it was already displayed above
    rendered_history = st.session_state.get("messages", [])
    if active_prompt and len(rendered_history) >= 2:
        history_to_display = rendered_history[:-2]
    else:
        history_to_display = rendered_history

    for msg in reversed(history_to_display):
        avatar = "🩺" if msg.get("role") == "assistant" else "👤"
        with st.chat_message(msg.get("role"), avatar=avatar):
            if msg.get("is_card"):
                st.markdown(
                    f'<div class="chat-response-card">{msg.get("content")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg.get("content"))
