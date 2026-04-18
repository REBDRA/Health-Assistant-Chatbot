import json
import os
import random
import streamlit as st
from dotenv import load_dotenv

from ai_service import HealthAIFacade

# 1. Page config MUST be the very first Streamlit command
st.set_page_config(page_title="Health Assistant AI", page_icon="🩺", layout="wide")

load_dotenv()


# ⭐ Rating stars
def get_stars(rating: str) -> str:
    try:
        num = float(rating.split("/")[0])
        full = int(num)
        half = 1 if num - full >= 0.5 else 0
        return "⭐" * full + (" ✨" if half else "")
    except (ValueError, AttributeError, IndexError):
        return "⭐⭐⭐⭐"


# 💧 Water Tracker Persistence Functions
WATER_FILE = "water_data.json"


def load_water_progress():
    if os.path.exists(WATER_FILE):
        try:
            with open(WATER_FILE, "r") as f:
                return json.load(f).get("water_litres", 0.0)
        except Exception:
            return 0.0
    return 0.0


def save_water_progress(amount):
    try:
        with open(WATER_FILE, "w") as f:
            json.dump({"water_litres": amount}, f)
    except Exception:
        pass


# 🎨 CSS styling & Fixed Footer
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
[data-testid="block-container"] { padding-bottom: 30px; } 

/* Aesthetic Card CSS (Chat Bubbles) */
.playful-card {
    background: rgba(255, 255, 255, 0.08); 
    backdrop-filter: blur(10px); 
    border: 1px solid rgba(137, 247, 254, 0.3);
    border-radius: 20px; 
    padding: 20px;
    color: #f1f1f1; 
    font-family: 'Nunito', sans-serif; 
    white-space: pre-wrap; 
    line-height: 1.6;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1); 
    transition: all 0.3s ease-in-out; 
    margin-bottom: 15px;
}
.playful-card:hover {
    transform: translateY(-6px); 
    box-shadow: 0px 12px 30px rgba(137, 247, 254, 0.2); 
    border-color: rgba(137, 247, 254, 0.8);
    background: rgba(255, 255, 255, 0.12);
}

/* 💎 BRUTE-FORCE DARK BORDERS FOR SIDEBAR WIDGETS 💎 */
.stApp [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(15, 32, 39, 0.75) !important; /* Dark chat-box background */
    background-image: none !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(137, 247, 254, 0.4) !important; /* Glowing blue border */
    border-radius: 15px !important; 
    box-shadow: 0px 10px 40px rgba(0, 0, 0, 0.5) !important; /* Deep shadow */
    transition: all 0.3s ease-in-out !important;
}
.stApp [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(137, 247, 254, 0.9) !important;
    box-shadow: 0px 12px 45px rgba(137, 247, 254, 0.2) !important;
}

/* Updated Button Aesthetic */
div[data-testid="stButton"] button {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(137, 247, 254, 0.3) !important;
    color: #f1f1f1 !important;
    font-family: 'Nunito', sans-serif !important;
    border-radius: 12px !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1) !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-4px) !important; 
    box-shadow: 0px 8px 20px rgba(137, 247, 254, 0.25) !important;
    border-color: rgba(137, 247, 254, 0.8) !important;
    background: rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
}

/* 💎 GEMINI-STYLE FLOATING CHAT INPUT 💎 */
div[data-testid="stChatInput"] {
    position: sticky !important;
    bottom: 30px !important;
    z-index: 9999 !important;
    background: rgba(15, 32, 39, 0.75) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(137, 247, 254, 0.4) !important;
    box-shadow: 0px 10px 40px rgba(0, 0, 0, 0.5) !important;
    padding: 5px !important;
}

/* Custom Fixed Footer for Copyright */
.custom-footer {
    position: fixed;
    bottom: 5px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'Nunito', sans-serif;
    z-index: 999999;
    pointer-events: none;
    text-align: center;
}
</style>

<div class="custom-footer">
    © 2026 Made with ❤️ by <b>Arpan</b>
</div>
""",
    unsafe_allow_html=True,
)

# 🔒 Secure API Key Loading
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing! Please set it in secrets.toml or as an environment variable."
    )
    st.stop()

SYSTEM_PROMPT = """
You are a highly intelligent and strict Medical Triage & Health AI.

You MUST return ONLY valid JSON in this format:
{
  "is_valid_query": true/false,
  "query_type": "symptom_triage" or "general_health",
  "direct_answer": "...",
  "remedies": ["...", "...", "..."],
  "advice": "...",
  "doctors": [
    {"name": "Dr. Firstname Lastname (Specialty)", "phone": "...", "location": "...", "rating": "..."}
  ],
  "error_message": "..."
}

CRITICAL RULES FOR VALIDATION & QUERY TYPE:
1. ANATOMICAL LOGIC: If the user states a contradiction, gibberish, or a non-health related joke -> set "is_valid_query": false and provide a polite "error_message" asking for clarity.
2. GENERAL HEALTH (e.g., "how much caffeine daily?", "what are vitamins?"): Set "query_type": "general_health". Provide the answer directly in "direct_answer" and additional tips in "advice". Leave "remedies" and "doctors" as empty arrays [].
3. SYMPTOMS/TRIAGE (e.g., "my head hurts", "I have a fever"): Set "query_type": "symptom_triage". Leave "direct_answer" empty "". Fill out "remedies", "advice", and provide EXACTLY 3 "doctors".

RULES FOR CONTENT GENERATION:
1. REMEDIES: Provide specific cures, solutions, or home remedies.
2. ADVICE: Provide broader health advice, lifestyle tips, or preventative measures.
3. DOCTOR GENERATION (CRITICAL FOR TRIAGE):
   - Identify the exact medical SPECIALTY needed for the user's symptom.
   - Include this specialty in brackets next to the name: e.g., "Dr. Amit Sharma (Ophthalmologist)".
   - DO NOT repeat the same doctor names for different queries. Generate distinct, varied, and realistic doctor names for West Bengal, India.
   - Give EXACTLY 3 doctors. Ratings must be 'X.X/5'.
"""

health_ai = HealthAIFacade(api_key=api_key)

# ==========================================
# 📐 NEW LAYOUT: 3 Columns
# ==========================================
left_col, main_col, right_col = st.columns([1, 2.2, 1], gap="large")

# ------------------------------------------
# ⚡ LEFT COLUMN: Quick Tools
# ------------------------------------------
with left_col:
    st.markdown("### ⚡ Quick Tools")

    with st.container(border=True):
        st.markdown("#### ⚖️ BMI Calculator")
        weight = st.number_input("Weight (kg)", min_value=10.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=50.0, value=170.0, step=1.0)

        if st.button("Calculate BMI", use_container_width=True):
            bmi = weight / ((height / 100) ** 2)
            if bmi < 18.5:
                status, color = "Underweight", "🔵"
            elif 18.5 <= bmi < 24.9:
                status, color = "Normal", "🟢"
            elif 25 <= bmi < 29.9:
                status, color = "Overweight", "🟠"
            else:
                status, color = "Obese", "🔴"
            st.success(f"**BMI: {bmi:.1f}**\n\n{color} {status}")

    # 💬 Chat Controls
    with st.container(border=True):
        st.markdown("#### 💬 Chat Controls")
        col_undo, col_clear = st.columns(2)

        if col_undo.button(
            "⏪ Undo Last", help="Remove your last message", use_container_width=True
        ):
            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                st.session_state.messages = st.session_state.messages[:-2]
                st.rerun()

        if col_clear.button(
            "🗑️ Clear All", help="Start a fresh chat", use_container_width=True
        ):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! Describe your symptoms or ask a general health question.",
                    "is_card": False,
                }
            ]
            st.rerun()

# ------------------------------------------
# 💡 RIGHT COLUMN: Wellness Hub
# ------------------------------------------
with right_col:
    st.markdown("### 💡 Wellness Hub")

    if "water_litres" not in st.session_state:
        st.session_state.water_litres = load_water_progress()

    with st.container(border=True):
        st.markdown("#### 💧 Water Tracker")

        progress_val = min(st.session_state.water_litres / 2.0, 1.0)
        st.progress(
            progress_val, text=f"{st.session_state.water_litres:.2f} / 2.0 Litres"
        )

        col1, col2, col3 = st.columns(3)
        if col1.button("➕ Drink", help="Add 0.25L", use_container_width=True):
            if st.session_state.water_litres < 2.0:
                st.session_state.water_litres = round(
                    st.session_state.water_litres + 0.25, 2
                )
                save_water_progress(st.session_state.water_litres)
                st.rerun()
        if col2.button("➖ Undo", help="Remove 0.25L", use_container_width=True):
            if st.session_state.water_litres >= 0.25:
                st.session_state.water_litres = round(
                    st.session_state.water_litres - 0.25, 2
                )
                save_water_progress(st.session_state.water_litres)
                st.rerun()
        if col3.button("🔄 Reset", use_container_width=True):
            st.session_state.water_litres = 0.0
            save_water_progress(0.0)
            st.rerun()

    with st.container(border=True):
        st.markdown("#### 🍎 Daily Tip")
        tips = [
            "Take a 5-minute walking break every hour.",
            "Screen time? Follow the 20-20-20 rule to rest your eyes.",
            "Aim for 7-8 hours of sleep for optimal immune function.",
            "Include a source of protein in every meal.",
        ]
        st.info(random.choice(tips))

# ------------------------------------------
# 🤖 MAIN COLUMN: The Chatbot Interface
# ------------------------------------------
with main_col:
    AI_AVATAR = "🤖"

    col_img, col_title = st.columns([1, 4])
    with col_img:
        st.markdown(
            f"<div style='font-size: 60px; text-align: center; margin-top: 10px;'>{AI_AVATAR}</div>",
            unsafe_allow_html=True,
        )
    with col_title:
        st.title("Health Assistant")

    st.info(
        "👋 Hi! Tell me what's bothering you, or ask me a health question — I’ll help you out 💙"
    )

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Describe your symptoms or ask a general health question.",
                "is_card": False,
            }
        ]

    # Display chat history naturally (No scrolling container)
    for msg in st.session_state.messages:
        avatar = AI_AVATAR if msg["role"] == "assistant" else "👤"

        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("is_card"):
                st.markdown(
                    f'<div class="playful-card">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

    # Input processing (Sticky floating CSS handles the bottom pinning)
    if prompt := st.chat_input("Describe your symptoms or ask a health question..."):
        current_history = list(st.session_state.messages)

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "is_card": False}
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.spinner("Analyzing..."):
                try:
                    data = health_ai.get_structured_response(
                        user_prompt=prompt, chat_history=current_history
                    )

                    if not data.get("is_valid_query", True):
                        output = data.get(
                            "error_message",
                            "I didn't quite understand that. Could you clarify?",
                        )
                        is_card = False
                    else:
                        is_card = True
                        output = ""

                        if data.get("query_type") == "general_health":
                            direct_answer = data.get("direct_answer", "")
                            advice = data.get("advice", "")
                            if direct_answer:
                                output += f"🩺 **Health Answer:**\n{direct_answer}\n"
                            if advice:
                                output += f"\n💡 **Additional Advice:**\n{advice}\n"

                        else:
                            remedies = data.get("remedies", [])
                            if remedies:
                                remedies_text = "\n".join(
                                    [f"{i}. {r}" for i, r in enumerate(remedies, 1)]
                                )
                                output += f"🌿 **Home Remedies & Recovery Steps:**\n{remedies_text}\n"

                            advice = data.get("advice", "")
                            if advice:
                                output += f"\n💡 **General Health Advice:**\n{advice}\n"

                            doctors = data.get("doctors", [])
                            if doctors:
                                output += "\n👨‍⚕️ **Recommended Doctors Near You:**\n\n"
                                for doc in doctors:
                                    stars = get_stars(doc.get("rating", ""))
                                    output += (
                                        f"🧑‍⚕️ **{doc.get('name', 'Unknown')}**\n"
                                        f"📍 {doc.get('location', 'Unknown')}\n"
                                        f"📞 {doc.get('phone', 'N/A')}\n"
                                        f"⭐ {stars}\n\n---\n\n"
                                    )

                            output += "\n*Disclaimer: I am an AI, not a doctor. Please consult a professional for medical emergencies.*"

                    if is_card:
                        st.markdown(
                            f'<div class="playful-card">{output}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(output)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": output, "is_card": is_card}
                    )

                except Exception as e:
                    st.error(f"System Error: {e}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Something went wrong. Try again.",
                            "is_card": False,
                        }
                    )
