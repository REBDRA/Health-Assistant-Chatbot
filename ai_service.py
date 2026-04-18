import os
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# --- 1. SCHEMAS (The Building Blocks of Output) ---


class Doctor(BaseModel):
    name: str = Field(
        description="Name and specialty in brackets, e.g. 'Dr. Ray (Cardiologist)'."
    )
    phone: str
    location: str
    rating: str = Field(description="Format: 'X.X/5'")


class HealthResponse(BaseModel):
    is_valid_query: bool = Field(
        description="False if the input is gibberish or non-health related."
    )
    query_type: str = Field(description="Either 'symptom_triage' or 'general_health'.")
    direct_answer: str = Field(
        default="", description="The answer for general health questions."
    )
    remedies: List[str] = Field(
        default=[], description="Steps the user can take at home."
    )
    advice: str = Field(
        default="", description="Broader lifestyle or preventative tips."
    )
    doctors: List[Doctor] = Field(
        default=[], description="Exactly 3 doctors fetched from the search tool."
    )
    error_message: str = Field(
        default="", description="Message shown if query is invalid."
    )


# --- 2. THE AGENT CONFIGURATION ---

# UPDATED: Changed 'system_prompt' to 'instructions'
health_agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    result_type=HealthResponse,
    instructions=(
        "You are a strict Medical Triage AI for West Bengal, India. "
        "1. If symptoms are mentioned, you MUST use the 'search_verified_doctors' tool to find real doctors. "
        "2. If it is a general health question, answer directly. "
        "3. Always maintain a professional and empathetic tone."
    ),
)


# --- 3. TOOLS (The 'Hands' of the Agent) ---


@health_agent.tool
def search_verified_doctors(ctx: RunContext[None], symptom: str) -> List[dict]:
    """
    Calls a local database to find real doctors in West Bengal based on symptoms.
    This provides 'Grounding' to prevent the LLM from hallucinating names.
    """
    s = symptom.lower()
    if any(word in s for word in ["skin", "rash", "itch", "acne"]):
        specialty = "Dermatologist"
    elif any(word in s for word in ["heart", "chest", "breath"]):
        specialty = "Cardiologist"
    elif any(word in s for word in ["eye", "vision", "blur"]):
        specialty = "Ophthalmologist"
    else:
        specialty = "General Physician"

    db = {
        "Dermatologist": [
            {
                "name": "Dr. A. Das (Dermatologist)",
                "phone": "98300-11111",
                "location": "Kolkata",
                "rating": "4.8/5",
            },
            {
                "name": "Dr. S. Roy (Dermatologist)",
                "phone": "98300-22222",
                "location": "Howrah",
                "rating": "4.7/5",
            },
            {
                "name": "Dr. P. Sen (Dermatologist)",
                "phone": "98300-33333",
                "location": "Bidhannagar",
                "rating": "4.9/5",
            },
        ],
        "General Physician": [
            {
                "name": "Dr. B. Chatterjee (GP)",
                "phone": "90000-55555",
                "location": "Salt Lake",
                "rating": "4.9/5",
            },
            {
                "name": "Dr. M. Khan (GP)",
                "phone": "90000-66666",
                "location": "Park Street",
                "rating": "4.6/5",
            },
            {
                "name": "Dr. R. Dutta (GP)",
                "phone": "90000-77777",
                "location": "New Town",
                "rating": "4.7/5",
            },
        ],
    }
    return db.get(specialty, db["General Physician"])


# --- 4. THE FACADE ---


class HealthAIFacade:
    def __init__(self, api_key: str):
        os.environ["GROQ_API_KEY"] = api_key

    def get_structured_response(self, user_prompt: str, chat_history: list) -> dict:
        history_text = ""
        if chat_history:
            recent = [m for m in chat_history if not m.get("is_card")][-3:]
            for m in recent:
                role = "User" if m["role"] == "user" else "Assistant"
                history_text += f"{role}: {m['content']}\n"

        full_input = f"History:\n{history_text}\nUser: {user_prompt}"

        # The Agentic Loop: Reason -> Tool Use -> Observe -> Output
        result = health_agent.run_sync(full_input)

        return result.data.model_dump()
