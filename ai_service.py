import os
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from duckduckgo_search import DDGS  # 100% Free Live Search


# --- 1. SCHEMAS ---
class Doctor(BaseModel):
    name: str = Field(
        description="Actual name of the doctor or clinic found in search."
    )
    phone: str = Field(description="Contact number or 'Visit Website' if not found.")
    location: str = Field(description="Specific area or address in West Bengal.")
    rating: str = Field(
        description="Rating from search results, or 'Verified' if found."
    )


class HealthResponse(BaseModel):
    is_valid_query: bool
    query_type: str
    direct_answer: str = ""
    remedies: List[str] = []
    advice: str = ""
    doctors: List[Doctor] = []
    error_message: str = ""


# --- 2. THE AGENT CONFIGURATION ---
health_agent = Agent(model="groq:llama-3.3-70b-versatile")


@health_agent.system_prompt
def add_health_instructions() -> str:
    return (
        "You are a strict Medical Triage & Health AI for West Bengal, India.\n\n"
        "CRITICAL RULES FOR QUERY TYPE:\n"
        "1. GENERAL HEALTH (e.g., 'how much caffeine daily?', 'what are vitamins?'): "
        "Set query_type='general_health'. Provide the answer in direct_answer and tips in advice. "
        "Leave remedies and doctors as empty arrays.\n"
        "2. SYMPTOMS/TRIAGE (e.g., 'my head hurts', 'red eyes', 'I have a fever'): "
        "Set query_type='symptom_triage'. Leave direct_answer empty. "
        "Fill out remedies, advice, AND you MUST use the 'get_live_doctors' tool to find real doctors, "
        "then provide EXACTLY 3 doctors.\n"
        "3. If the query is nonsense or non-health related, set is_valid_query=false with an error_message.\n\n"
        "RULES FOR DOCTORS (CRITICAL FOR TRIAGE):\n"
        "- You MUST call the 'get_live_doctors' tool for ANY symptom query. This is mandatory.\n"
        "- Extract real clinic/doctor names, phone numbers, and addresses from the search results.\n"
        "- Do NOT invent doctor information. Only use data from search results.\n"
        "- Identify the exact medical SPECIALTY needed (e.g., Ophthalmologist for eye issues).\n"
        "- Include the specialty in the name: e.g., 'Dr. Amit Sharma (Ophthalmologist)'.\n"
        "- Ratings must be in 'X.X/5' format.\n\n"
        "RULES FOR CONTENT:\n"
        "- remedies: Provide specific cures, solutions, or home remedies.\n"
        "- advice: Provide broader health advice, lifestyle tips, or preventative measures."
    )


# --- 3. THE FREE SEARCH TOOL (Grounding Building Block) ---
@health_agent.tool
def get_live_doctors(ctx: RunContext[None], symptom: str) -> str:
    """Finds real practicing doctors and clinics in West Bengal via live web search."""
    # We build a hyper-local query
    query = f"best {symptom} specialists and clinics in Kolkata West Bengal phone number and address"

    with DDGS() as ddgs:
        # Fetch top 5 snippets from the live web
        results = [r for r in ddgs.text(query, max_results=5)]

    return str(results)


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

        # Building Block: The Agentic Loop
        # The agent reasons -> calls get_live_doctors -> observes web data -> generates JSON
        result = health_agent.run_sync(full_input, output_type=HealthResponse)
        return result.output.model_dump()
