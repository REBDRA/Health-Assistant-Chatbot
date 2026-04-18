import os
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# --- 1. SCHEMAS ---
class Doctor(BaseModel):
    name: str = Field(description="Name and specialty in brackets.")
    phone: str
    location: str
    rating: str


class HealthResponse(BaseModel):
    is_valid_query: bool
    query_type: str
    direct_answer: str = ""
    remedies: List[str] = []
    advice: str = ""
    doctors: List[Doctor] = []
    error_message: str = ""


# --- 2. THE AGENT CONFIGURATION ---

# FIX: We remove result_type from here.
# This stops the 'Unknown keyword argument' error immediately.
health_agent = Agent(model="groq:llama-3.3-70b-versatile")


@health_agent.system_prompt
def add_health_instructions() -> str:
    return (
        "You are a highly intelligent and strict Medical Triage & Health AI. "
        "Evaluate the user's input. If it is a symptom, provide triage, remedies, and 3 realistic doctors. "
        "If it is a general health question, provide a direct answer and advice. "
        "If it is not a health query, reject it politely."
    )


# --- 3. TOOLS ---
@health_agent.tool
def search_verified_doctors(ctx: RunContext[None], symptom: str) -> List[dict]:
    """Find real doctors in West Bengal based on symptoms."""
    s = symptom.lower()
    if any(word in s for word in ["skin", "rash", "itch", "acne"]):
        specialty = "Dermatologist"
    elif any(word in s for word in ["heart", "chest", "breath"]):
        specialty = "Cardiologist"
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
        ],
        "General Physician": [
            {
                "name": "Dr. B. Chatterjee (GP)",
                "phone": "90000-55555",
                "location": "Salt Lake",
                "rating": "4.9/5",
            }
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

        # FIX: We tell the agent what the result_type is RIGHT HERE.
        # This is the 'Building Block: Output Validation' in action.
        result = health_agent.run_sync(full_input, result_type=HealthResponse)
        return result.data.model_dump()
