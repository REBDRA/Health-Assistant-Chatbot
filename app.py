# type: ignore
import json
import os
import random
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Import the new facade from your ai_service module
from ai_service import HealthAIFacade

# 1. Page config MUST be the very first Streamlit command
st.set_page_config(page_title="Health Assistant AI", page_icon="🩺", layout="wide")

# Load env
load_dotenv()


# ⭐ Rating stars - Optimized exception handling
def get_stars(rating: str) -> str:
    try:
        num = float(rating.split("/")[0])
        full = int(num)
        half = 1 if num - full >= 0.5 else 0
        return "⭐" * full + (" ✨" if half else "")
    except ValueError, AttributeError, IndexError:
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
[data-testid="block-container"] { padding-bottom: 150px; }
[data-testid="stChatMessage"] { border-radius: 15px; padding: 10px; transition: 0.3s; }
[data-testid="stChatMessage"]:hover { transform: scale(1.01); background: rgba(255, 255, 255, 0.03); }
.playful-card {
    background: rgba(255, 255, 255, 0.08); 
    backdrop-filter: blur(10px); 
    border: 2px dashed rgba(137, 247, 254, 0.5); 
    border-radius: 20px; 
    padding: 20px;
    color: #f1f1f1; 
    font-family: 'Nunito', 'Comic Sans MS', sans-serif; 
    white-space: pre-wrap; 
    line-height: 1.6;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.playful-card:hover {
    transform: translateY(-5px) rotate(0.5deg); 
    box-shadow: 0px 10px 25px rgba(137, 247, 254, 0.15); 
    border-color: #89f7fe;
    background: rgba(255, 255, 255, 0.12);
}

/* Custom Fixed Footer for Copyright */
.custom-footer {
    position: fixed;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(255, 255, 255, 0.6);
    font-size: 13px;
    font-family: 'Nunito', sans-serif;
    z-index: 999999;
    pointer-events: none;
    text-align: center;
}

/* Add slight padding to bottom block to ensure input box doesn't cover footer */
[data-testid="stBottomBlockContainer"] {
    padding-bottom: 35px !important;
}
</style>

<div class="custom-footer">
    © 2026 Made with ❤️ by <b>Arpan</b>
</div>
""",
    unsafe_allow_html=True,
)

# 🔒 Secure API Key Loading (Compact & Safe)
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing! Please set it in secrets.toml or as an environment variable."
    )
    st.stop()  # Prevents the app from crashing later when initializing the client

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

# Initialize Groq Client and the new HealthAIFacade
client = Groq(api_key=api_key)
health_ai = HealthAIFacade(client=client, system_prompt=SYSTEM_PROMPT)


# ==========================================
# 📐 NEW LAYOUT: 3 Columns
# ==========================================
left_col, main_col, right_col = st.columns([1, 2.2, 1], gap="large")

# ------------------------------------------
# ⚡ LEFT COLUMN: Quick Tools
# ------------------------------------------
with left_col:
    st.markdown("### ⚡ Quick Tools")

    # BMI Calculator Widget
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

# ------------------------------------------
# 💡 RIGHT COLUMN: Wellness Hub
# ------------------------------------------
with right_col:
    st.markdown("### 💡 Wellness Hub")

    # Interactive Water Tracker
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

    # Daily Tip
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
# ------------------------------------------
# 🤖 MAIN COLUMN: The Chatbot Interface
# ------------------------------------------
with main_col:
    # 🖼️ Cool Robot Avatar
    AI_AVATAR = "🤖"

    col_img, col_title = st.columns([1, 4])
    with col_img:
        # FIX: Render the emoji as large text instead of using st.image()
        st.markdown(
            f"<div style='font-size: 60px; text-align: center; margin-top: 10px;'>{AI_AVATAR}</div>",
            unsafe_allow_html=True,
        )
    with col_title:
        st.title("Health Assistant")

    # 💙 Welcome
    st.info(
        "👋 Hi! Tell me what's bothering you, or ask me a health question — I’ll help you out 💙"
    )

    # Chat history initialization...

    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Describe your symptoms or ask a general health question.",
                "is_card": False,
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        avatar = AI_AVATAR_URL if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("is_card"):
                st.markdown(
                    f'<div class="playful-card">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

    # Input processing
    if prompt := st.chat_input("Describe your symptoms or ask a health question..."):
        # We save the current history to pass to the facade (excluding the prompt we are about to add)
        # We do this because your facade explicitly appends the new `user_prompt` inside `get_structured_response`.
        current_history = list(st.session_state.messages)

        # Append user prompt to state
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "is_card": False}
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR_URL):
            with st.spinner("Analyzing..."):
                try:
                    # 🚀 Call the Facade instead of raw API
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

                        # --- BRANCH 1: General Health Questions ---
                        if data.get("query_type") == "general_health":
                            direct_answer = data.get("direct_answer", "")
                            advice = data.get("advice", "")

                            if direct_answer:
                                output += f"🩺 **Health Answer:**\n{direct_answer}\n"
                            if advice:
                                output += f"\n💡 **Additional Advice:**\n{advice}\n"

                        # --- BRANCH 2: Symptom Triage & Remedies ---
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

                    # Render to screen
                    if is_card:
                        st.markdown(
                            f'<div class="playful-card">{output}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(output)

                    # Save to session state so it survives reloads
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
