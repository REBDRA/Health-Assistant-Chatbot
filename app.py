import streamlit as st
import os
from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Health Assistant AI", page_icon="🩺", layout="centered")
st.title("🩺 Health Assistant AI")

if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
    st.warning("⚠️ GROQ_API_KEY is not set. Please add it to your .env file to continue.")
    st.stop()

# Initialize the Groq client and patch it with Instructor
try:
    # instructor patches the groq client to natively output strictly via Pydantic
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    client = instructor.from_groq(client, mode=instructor.Mode.JSON)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# ----------------- PYDANTIC SCHEMAS -----------------
class Doctor(BaseModel):
    name: str = Field(description="Doctor's Full Name")
    phone: str = Field(description="Phone Number")
    location: str = Field(description="Specific Area/City")
    rating: str = Field(description="X/5 Stars")

class HealthResponse(BaseModel):
    is_valid_query: bool = Field(
        description="True if the user described a medical pain/symptom. False if gibberish, non-medical, or making no sense."
    )
    remedies: Optional[List[str]] = Field(
        default=None, 
        description="3 actionable home remedies or recovery steps. Required if is_valid_query is True."
    )
    doctors: Optional[List[Doctor]] = Field(
        default=None, 
        description="3 relevant medical professionals. Required if is_valid_query is True."
    )
    error_message: Optional[str] = Field(
        default=None, 
        description="If is_valid_query is False, exact phrase: 'I'm sorry, I couldn't understand that. Please describe the area of your body where you are experiencing pain so I can assist you.'"
    )

class finalOutput(BaseModel):
    response: HealthResponse
# ----------------------------------------------------

SYSTEM_PROMPT = """You are a professional Health Assistant AI. Your strict job is to output structured data according to the provided schema based on these rules:
Rule 1: If the user describes pain in a body part, set is_valid_query to true, provide exactly 3 actionable home remedies, and list exactly 3 relevant doctors.
Rule 2: If the input is gibberish, non-medical, or makes no sense, set is_valid_query to false and provide the exact error_message."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your professional Health Assistant AI. Please describe the area of your body where you are experiencing pain or your medical symptoms."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your symptoms via structured reasoning..."):
            messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Pass recent history
            messages_payload += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-3:] if m["role"] == "user"]
            
            try:
                # Instructor guarantees a HealthResponse structured exactly as defined above
                response_data: HealthResponse = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=messages_payload,
                    response_model=HealthResponse,
                    temperature=0.0 # Deterministic strictly for parsing
                )
                
                # Format the response back into perfect Markdown string exactly matching the gates
                if not response_data.is_valid_query:
                    final_text = response_data.error_message or "I'm sorry, I couldn't understand that. Please describe the area of your body where you are experiencing pain so I can assist you."
                else:
                    final_text = "**Home Remedies & Recovery Steps:**\n"
                    if response_data.remedies:
                        for idx, remedy in enumerate(response_data.remedies, 1):
                            final_text += f"{idx}. {remedy}\n"
                    
                    final_text += "\n**Recommended Doctors:**\n\n"
                    if response_data.doctors:
                        for doc in response_data.doctors:
                            final_text += f"**Name:** {doc.name}\n\n**Phone:** {doc.phone}\n\n**Location:** {doc.location}\n\n**Rating:** {doc.rating}\n\n---\n"
                
                # Apply standard system disclaimer to all outputs, as requested
                final_text += "\n\n*Disclaimer: I am an AI, not a doctor. In case of a life-threatening emergency, please call your local emergency services immediately.*"

                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                st.error(f"API Error during structured output extraction: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I encountered an error connecting to my intelligence systems. Please try again."})
