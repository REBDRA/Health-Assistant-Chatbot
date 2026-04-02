# type:ignore
import json
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# 1. Page config MUST be the very first Streamlit command
st.set_page_config(page_title="Health Assistant AI", page_icon="🩺")

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


st.title("🩺 Health Assistant AI")

# 🎨 CSS styling (Kept exactly as you designed it)
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
</style>
""",
    unsafe_allow_html=True,
)

# 🖼️ Image
st.image(
    "[https://cdn-icons-png.flaticon.com/512/3774/3774299.png](https://cdn-icons-png.flaticon.com/512/3774/3774299.png)",
    width=120,
)

# 🔒 Secure API Key Loading (Compact & Safe)
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing! Please set it in secrets.toml or as an environment variable."
    )
    st.stop()  # Prevents the app from crashing later when initializing the client

client = Groq(api_key=api_key)
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

# 💙 Welcome
st.info("👋 Hi! Tell me what's bothering you — I’ll help you feel better 💙")

# Chat history initialization (Added 'is_card' state)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Describe your symptoms.",
            "is_card": False,
        }
    ]

# Display chat history (Fixes the disappearing CSS bug on reload)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_card"):
            st.markdown(
                f'<div class="playful-card">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# Input processing
if prompt := st.chat_input("Describe your symptoms..."):
    # Append user prompt to history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "is_card": False}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # 🚀 Native JSON Mode for absolute stability
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={
                        "type": "json_object"
                    },  # Forces LLM to output clean JSON
                )

                # Parse JSON directly (no string replacements needed)
                data = json.loads(response.choices[0].message.content)

                if not data.get("is_valid_query", True):
                    output = data.get(
                        "error_message",
                        "I didn't quite understand that. Could you clarify?",
                    )
                    is_card = False
                else:
                    is_card = True
                    # Compact string building
                    remedies_text = "\n".join(
                        [f"{i}. {r}" for i, r in enumerate(data.get("remedies", []), 1)]
                    )
                    advice = data.get("advice", "")

                    output = (
                        f"🌿 **Home Remedies & Recovery Steps:**\n{remedies_text}\n"
                    )

                    if advice:
                        output += f"\n💡 **General Health Advice:**\n{advice}\n"

                    output += "\n👨‍⚕️ **Recommended Doctors Near You:**\n\n"

                    for doc in data.get("doctors", []):
                        stars = get_stars(doc.get("rating", ""))
                        output += (
                            f"🧑‍⚕️ **{doc.get('name', 'Unknown')}**\n"
                            f"📍 {doc.get('location', 'Unknown')}\n"
                            f"📞 {doc.get('phone', 'N/A')}\n"
                            f"⭐ {stars}\n\n---\n\n"
                        )

                    output += "\n*Disclaimer: I am an AI, not a doctor.*"

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
                # Logging the actual error makes debugging much easier
                st.error(f"System Error: {e}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Something went wrong. Try again.",
                        "is_card": False,
                    }
                )
