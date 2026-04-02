# type:ignore
import json
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load env
load_dotenv()


# ⭐ Rating stars
def get_stars(rating):
    try:
        num = float(rating.split("/")[0])
        full = int(num)
        half = 1 if num - full >= 0.5 else 0
        return "⭐" * full + (" ✨" if half else "")
    except:
        return "⭐⭐⭐⭐"


st.set_page_config(page_title="Health Assistant AI", page_icon="🩺")

st.title("🩺 Health Assistant AI")

# 🎨 CSS styling
st.markdown(
    """
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* FIX: Scroll stickiness at the bottom */
[data-testid="block-container"] {
    padding-bottom: 150px; 
}

/* Streamlit Chat Message Hover */
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    transition: 0.3s;
}
[data-testid="stChatMessage"]:hover {
    transform: scale(1.01);
    background: rgba(255, 255, 255, 0.03);
}

/* Playful Card Styling */
.playful-card {
    background: rgba(255, 255, 255, 0.08); /* Semi-transparent to match background */
    backdrop-filter: blur(10px); /* Glassmorphism effect */
    border: 2px dashed rgba(137, 247, 254, 0.5); /* Playful dashed border */
    border-radius: 20px; 
    padding: 20px;
    color: #f1f1f1; /* Light text for dark mode */
    font-family: 'Nunito', 'Comic Sans MS', sans-serif; 
    white-space: pre-wrap; /* Ensures spacing and newlines render properly */
    line-height: 1.6;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy hover */
}

.playful-card:hover {
    transform: translateY(-5px) rotate(0.5deg); /* Bounce and tilt */
    box-shadow: 0px 10px 25px rgba(137, 247, 254, 0.15); /* Soft glowing shadow */
    border-color: #89f7fe;
    background: rgba(255, 255, 255, 0.12);
}
</style>
""",
    unsafe_allow_html=True,
)

# 🖼️ Image
st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=120)

# 🔒 Secure API Key Loading (Works locally and on Streamlit)
api_key = None

try:
    # First, try to fetch from Streamlit secrets
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    # This catches the StreamlitSecretNotFoundError (missing file)
    # AND KeyError (file exists but key doesn't)
    pass

# Fallback to standard environment variables if not found in st.secrets
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

# Optional: Add a check to warn you if it fails completely
if not api_key:
    st.error(
        "GROQ_API_KEY is missing! Please set it in secrets.toml or as an environment variable."
    )
else:
    client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are a highly intelligent and strict Medical Triage AI.

You MUST return ONLY valid JSON in this format:
{
  "is_valid_query": true/false,
  "remedies": ["...", "...", "..."],
  "doctors": [
    {"name": "Dr. Firstname Lastname (Specialty)", "phone": "...", "location": "...", "rating": "..."},
    {"name": "Dr. Firstname Lastname (Specialty)", "phone": "...", "location": "...", "rating": "..."},
    {"name": "Dr. Firstname Lastname (Specialty)", "phone": "...", "location": "...", "rating": "..."}
  ],
  "error_message": "..."
}

CRITICAL RULES FOR VALIDATION:
1. ANATOMICAL LOGIC: If the user states a contradiction (e.g., "headache in chest"), gibberish, or a joke -> set "is_valid_query": false and provide a polite "error_message" asking for clarity.
2. ONLY if the symptom makes sense -> set "is_valid_query": true.

RULES FOR DOCTOR GENERATION (CRITICAL):
1. First, identify the exact medical SPECIALTY needed for the user's symptom (e.g., Ophthalmologist for eyes, Neurologist for headaches, Dermatologist for skin).
2. You MUST include this specialty in brackets next to the name: e.g., "Dr. Amit Sharma (Ophthalmologist)".
3. DO NOT repeat the same doctor names for different queries. Generate distinct, varied, and realistic doctor names for West Bengal, India.
4. Give EXACTLY 3 doctors.
5. Ratings must be 'X.X/5'.
"""

# 💙 Welcome
st.info("👋 Hi! Tell me what's bothering you — I’ll help you feel better 💙")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Describe your symptoms."}
    ]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )

                raw = response.choices[0].message.content
                raw = raw.strip().replace("```json", "").replace("```", "").strip()

                data = json.loads(raw)

                if not data["is_valid_query"]:
                    output = data["error_message"]
                else:
                    output = "🌿 **Home Remedies & Recovery Steps:**\n"
                    for i, r in enumerate(data["remedies"], 1):
                        output += f"{i}. {r}\n"

                    output += "\n👨‍⚕️ **Recommended Doctors Near You:**\n\n"

                    for doc in data["doctors"]:
                        stars = get_stars(doc["rating"])

                        output += (
                            f"🧑‍⚕️ **{doc['name']}**\n"
                            f"📍 {doc['location']}\n"
                            f"📞 {doc['phone']}\n"
                            f"⭐ {stars}\n\n---\n\n"
                        )

                output += "\n*Disclaimer: I am an AI, not a doctor.*"

                # 🎨 Card UI (Now properly using the CSS class!)
                st.markdown(
                    f"""
                <div class="playful-card">
{output}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": output}
                )

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Something went wrong. Try again."}
                )
