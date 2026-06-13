import os
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from ddgs import DDGS  # 100% Free Live Search (renamed from duckduckgo_search)


# --- 1. SCHEMAS ---
class Doctor(BaseModel):
    name: str = Field(
        description="Actual name of the doctor or clinic found in search."
    )
    phone: str = Field(description="Contact number or 'Visit Website' if not found.")
    location: str = Field(
        description="Specific area or address near the user's location."
    )
    rating: str = Field(
        description="Rating from search results, or 'Verified' if found."
    )
    link: str = Field(
        description="URL to the doctor's profile or website from the search results, or empty if none.",
        default="",
    )


class HealthResponse(BaseModel):
    is_valid_query: bool
    query_type: str
    direct_answer: str = ""
    remedies: List[str] = Field(default_factory=list)
    advice: str = ""
    doctors: List[Doctor] = Field(default_factory=list)
    error_message: str = ""


def get_health_instructions() -> str:
    return (
        "You are a strict Medical Triage & Health AI.\n\n"
        "CRITICAL RULES FOR QUERY TYPE:\n"
        "1. GENERAL HEALTH (e.g., 'how much caffeine daily?', 'what are vitamins?'): "
        "Set query_type='general_health'. Provide the answer in direct_answer and tips in advice. "
        "Leave remedies and doctors as empty arrays.\n"
        "2. SYMPTOMS/TRIAGE (e.g., 'my head hurts', 'red eyes', 'I have a fever'): "
        "Set query_type='symptom_triage'. Leave direct_answer empty. "
        "Fill out remedies, advice. For doctors, use ONLY the real search results provided in the prompt — "
        "extract name, phone, location, rating from those results. Provide EXACTLY 3 doctors.\n"
        "3. If the query is nonsense or non-health related, set is_valid_query=false with an error_message.\n\n"
        "RULES FOR DOCTORS (CRITICAL FOR TRIAGE):\n"
        "- Extract real clinic/doctor names, phone numbers, addresses, and URL links from the search results given to you.\n"
        "- Do NOT invent doctor information. Only use data from the provided search results.\n"
        "- Identify the exact medical SPECIALTY needed (e.g., Ophthalmologist for eye issues).\n"
        "- Include the specialty in the name: e.g., 'Dr. Amit Sharma (Ophthalmologist)'.\n"
        "- Ratings must be in 'X.X/5' format. If none found, use '4.2/5'.\n\n"
        "RULES FOR CONTENT:\n"
        "- remedies: Provide specific cures, solutions, or home remedies.\n"
        "- advice: Provide broader health advice, lifestyle tips, or preventative measures."
    )


def create_health_agent() -> Agent:
    return Agent(
        model="groq:llama-3.3-70b-versatile",
        output_type=HealthResponse,
        system_prompt=get_health_instructions(),
    )


# --- 3. SEPARATE LIVE SEARCH FUNCTION (called explicitly in Python, not as agent tool) ---
def fetch_live_doctors(symptom: str, location: str) -> str:
    """Fetches real doctor info via DuckDuckGo and returns raw text for the agent to parse."""
    try:
        # Determine medical specialty for the symptom
        query = (
            f"best {symptom} specialist doctor clinic {location} contact phone address"
        )
        with DDGS() as d:
            results = list(d.text(query, max_results=6))
        if not results:
            return "No search results found."
        # Format results into readable text
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', '')}\n"
                f"Snippet: {r.get('body', '')}\n"
                f"URL: {r.get('href', '')}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


# --- 4. THE FACADE ---
class HealthAIFacade:
    def __init__(self, api_key: str):
        os.environ["GROQ_API_KEY"] = api_key
        self.health_agent = create_health_agent()

    def get_structured_response(
        self,
        user_prompt: str,
        chat_history: list,
        user_location: str = "Kolkata, West Bengal, India",
    ) -> dict:
        history_text = ""
        if chat_history:
            recent = [m for m in chat_history if not m.get("is_card")][-3:]
            for m in recent:
                role = "User" if m["role"] == "user" else "Assistant"
                history_text += f"{role}: {m['content']}\n"

        # Step 1: Quick classification — is this a symptom query?
        # We do a fast check by running a cheap classification first.
        classify_agent = Agent(
            model="groq:llama-3.3-70b-versatile",
            output_type=str,
        )
        classify_result = classify_agent.run_sync(
            f"Is this a symptom/medical complaint query? Reply only 'yes' or 'no': {user_prompt}"
        )
        is_symptom = "yes" in classify_result.output.lower()

        # Step 2: If symptom query, fetch live doctors BEFORE calling the main agent
        doctor_context = ""
        if is_symptom:
            raw_search = fetch_live_doctors(user_prompt, user_location)
            doctor_context = (
                f"\n\n=== LIVE SEARCH RESULTS FOR DOCTORS ===\n"
                f"{raw_search}\n"
                f"=== END OF SEARCH RESULTS ===\n\n"
                f"Using the search results above, extract EXACTLY 3 real doctors/clinics near {user_location}. "
                f"If the search results are sparse, use them as best you can and fill ratings as '4.2/5'."
            )

        full_input = (
            f"User Location: {user_location}\n"
            f"{'History:\n' + history_text if history_text else ''}"
            f"User Query: {user_prompt}"
            f"{doctor_context}"
        )

        # Step 3: Run the main structured-output agent with the pre-fetched context
        result = self.health_agent.run_sync(full_input)
        return result.output.model_dump()
