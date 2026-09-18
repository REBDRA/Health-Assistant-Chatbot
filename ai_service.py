import os
import re
import time
import logging
from functools import lru_cache
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.groq import GroqModel
from ddgs import DDGS

logger = logging.getLogger(__name__)

# --- 1. SCHEMAS ---

class Doctor(BaseModel):
    name: str = Field(
        description="Actual name of the doctor or clinic found in search."
    )
    phone: str = Field(
        default="Visit Website",
        description="Contact number or 'Visit Website' if phone not found."
    )
    location: str = Field(
        default="Not specified",
        description="Specific area, clinic address, or locality near the user's location."
    )
    rating: str = Field(
        default="4.5/5",
        description="Rating from search results (format X.X/5) or 'Verified'."
    )
    link: str = Field(
        default="",
        description="URL to the doctor's profile, clinic page, or search listing."
    )


class HealthResponse(BaseModel):
    is_valid_query: bool = Field(
        description="True if query is health/medical related, False for off-topic/nonsense."
    )
    query_type: str = Field(
        description="'symptom_triage' for physical/mental complaints, 'general_health' for educational health questions."
    )
    direct_answer: str = Field(
        default="",
        description="Comprehensive answer for general health queries or overview for symptoms."
    )
    remedies: List[str] = Field(
        default_factory=list,
        description="Actionable home remedies, recovery steps, or first-aid tips."
    )
    advice: str = Field(
        default="",
        description="General lifestyle, dietary, preventive guidance, or medical warnings."
    )
    doctors: List[Doctor] = Field(
        default_factory=list,
        description="List of real clinics or doctors extracted strictly from provided search results."
    )
    error_message: str = Field(
        default="",
        description="Helpful message explaining why query could not be processed if invalid."
    )


class QueryIntent(BaseModel):
    is_health_related: bool = Field(
        description="True if query pertains to health, medicine, fitness, symptoms, or biology."
    )
    is_symptom: bool = Field(
        description="True if user is describing an active illness, pain, bodily complaint, or injury."
    )
    specialty: str = Field(
        default="General Physician",
        description="The medical specialist best suited for this issue (e.g. Dermatologist, Cardiologist, ENT)."
    )
    search_keywords: str = Field(
        default="",
        description="2 to 4 keywords describing the condition for targeted clinic search (e.g. 'migraine headache neurology')."
    )


# --- 2. SYSTEM INSTRUCTIONS ---

HEALTH_SYSTEM_PROMPT = """You are a compassionate, certified-level Clinical Triage & Health Assistant AI.

Your role is to assess user queries with strict medical prudence, empathy, and evidence-based guidance.

CRITICAL PROTOCOLS:
1. QUERY TYPES:
   - 'general_health': For wellness, nutrition, fitness, medication explanations, or medical facts.
     Provide a clear, educational direct_answer and practical lifestyle tips in advice. Keep remedies and doctors empty.
   - 'symptom_triage': For active bodily or mental complaints (e.g., pain, rash, nausea, fever).
     - Provide immediate home remedies & safe self-care steps in 'remedies' (numbered or listed).
     - Provide broader preventive advice and warning signs/red flags in 'advice'.
     - For 'doctors', extract ONLY up to 3 genuine doctors or clinics from the provided search results below. If no search results or irrelevant, leave 'doctors' empty.
   - If completely off-topic (e.g. coding, finance, trivia) or gibberish, set is_valid_query=false and provide a polite redirect in error_message.

2. RULES FOR EXTRACTED DOCTORS:
   - Extract real clinic/doctor names, telephone/contact, locality, and URLs from the SEARCH RESULTS provided in prompt.
   - NEVER invent or fabricate imaginary doctor names. If the search results do not contain relevant clinics, return an empty doctors array.
   - Include the medical specialty in the name if known (e.g. 'Dr. Ananya Roy (Cardiologist)' or 'Apollo Clinic (Dermatology)').
   - Format ratings as 'X.X/5' or 'Verified'.

3. SAFETY & RED FLAGS:
   - If symptoms indicate life-threatening conditions (e.g. crushing chest pain, sudden numbness, severe difficulty breathing, uncontrollable bleeding), prominently urge immediate emergency services (e.g. call 112/911 or visit the nearest ER) at the start of advice.
"""

INTENT_SYSTEM_PROMPT = """You are an ultra-fast medical query parser.
Classify the user query and extract:
1. is_health_related: boolean
2. is_symptom: boolean (true if describing a bodily complaint, pain, or illness requiring diagnosis/triage)
3. specialty: most appropriate medical specialty (e.g., ENT Specialist, Orthopedic, Dermatologist, Neurologist, General Physician)
4. search_keywords: 2 to 4 concise search keywords for finding clinics (e.g. 'orthopedic joint pain')
"""


# --- 3. LIVE DOCTOR SEARCH ENGINE ---

def sanitize_search_term(text: str) -> str:
    """Removes special characters and limits query length to keep DDG search clean."""
    cleaned = re.sub(r"[^\w\s-]", " ", text)
    words = cleaned.strip().split()
    return " ".join(words[:6])


@lru_cache(maxsize=64)
def _cached_doctor_search(search_query: str) -> str:
    try:
        with DDGS(timeout=5) as d:
            results = list(d.text(search_query, max_results=5))
        if not results:
            return ""

        formatted = []
        for r in results:
            title = r.get("title", "").strip()
            snippet = r.get("body", "").strip()
            link = r.get("href", "").strip()
            if title or snippet:
                formatted.append(
                    f"Name/Clinic: {title}\nDetails: {snippet}\nWebsite: {link}"
                )
        return "\n\n---\n\n".join(formatted)
    except Exception as exc:
        logger.warning("DDGS search exception for '%s': %s", search_query, exc)
        return ""


def fetch_live_doctors(specialty: str, location: str, keywords: str = "") -> str:
    """Fetches real clinics and doctor listings using targeted query."""
    clean_loc = sanitize_search_term(location)
    clean_spec = sanitize_search_term(specialty)
    clean_kw = sanitize_search_term(keywords) if keywords else ""

    # Build targeted search query
    query = f"top {clean_spec} {clean_kw} clinic doctor in {clean_loc} contact address"
    query = " ".join(query.split())  # normalize spaces

    raw_results = _cached_doctor_search(query)
    if not raw_results and clean_loc:
        # Fallback to broader specialty search in location
        broad_query = f"{clean_spec} hospital clinic {clean_loc} phone"
        raw_results = _cached_doctor_search(broad_query)

    return raw_results


# --- 4. FACADE & ORCHESTRATION ---

class HealthAIFacade:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key cannot be empty.")

        self.api_key = api_key.strip()
        os.environ["GROQ_API_KEY"] = self.api_key

        self.provider = GroqProvider(api_key=self.api_key)

        # Fast tier: llama-3.1-8b-instant for sub-300ms classification & intent extraction
        self.fast_model = GroqModel("llama-3.1-8b-instant", provider=self.provider)
        self.intent_agent = Agent(
            model=self.fast_model,
            output_type=QueryIntent,
            system_prompt=INTENT_SYSTEM_PROMPT,
        )

        # Clinical tier: llama-3.3-70b-versatile for nuanced medical triage & structuring
        self.triage_model = GroqModel("llama-3.3-70b-versatile", provider=self.provider)
        self.health_agent = Agent(
            model=self.triage_model,
            output_type=HealthResponse,
            system_prompt=HEALTH_SYSTEM_PROMPT,
        )

    def validate_key(self) -> Tuple[bool, str]:
        """Validates the Groq API key with a minimal request."""
        try:
            test_agent = Agent(self.fast_model, output_type=str)
            test_agent.run_sync("ping", timeout=8)
            return True, "API Key is valid and active."
        except Exception as e:
            msg = str(e)
            if "invalid_api_key" in msg.lower() or "401" in msg:
                return False, "Invalid Groq API key. Please check your credentials."
            if "429" in msg or "rate_limit" in msg.lower():
                return False, "Groq API rate limit reached. Please wait a moment."
            return False, f"Connection error: {msg[:100]}"

    def _retry_run(self, agent: Agent, prompt: str, retries: int = 2):
        """Runs an agent with backoff retry on transient errors."""
        last_error = None
        for attempt in range(retries + 1):
            try:
                return agent.run_sync(prompt)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if ("429" in err_str or "rate limit" in err_str) and attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                if attempt == retries:
                    raise last_error

    def get_structured_response(
        self,
        user_prompt: str,
        chat_history: Optional[list] = None,
        user_location: str = "Kolkata, West Bengal, India",
    ) -> dict:
        """Processes user prompt and returns structured health recommendations."""
        # Format recent conversation context
        history_text = ""
        if chat_history:
            recent = [m for m in chat_history if not m.get("is_card")][-4:]
            for m in recent:
                role = "User" if m.get("role") == "user" else "Assistant"
                history_text += f"{role}: {m.get('content', '')}\n"

        # Tier 1: Fast Intent & Specialty Classification (sub-300ms)
        try:
            intent_result = self._retry_run(
                self.intent_agent,
                f"Classify query: '{user_prompt}'"
            )
            intent: QueryIntent = intent_result.output
        except Exception:
            # Fallback intent if fast classifier fails
            intent = QueryIntent(
                is_health_related=True,
                is_symptom=True,
                specialty="General Physician",
                search_keywords="medical doctor clinic"
            )

        # Quick reject off-topic non-health queries
        if not intent.is_health_related:
            return HealthResponse(
                is_valid_query=False,
                query_type="invalid",
                error_message=(
                    "I am an AI Health Assistant focused solely on health, medicine, and wellness. "
                    "Please ask me about symptoms, medical concerns, fitness, or nutrition!"
                ),
            ).model_dump()

        # Tier 2: Targeted Doctor Search for Symptom Queries
        doctor_context = ""
        if intent.is_symptom and user_location:
            raw_search = fetch_live_doctors(
                specialty=intent.specialty,
                location=user_location,
                keywords=intent.search_keywords,
            )
            if raw_search:
                doctor_context = (
                    f"\n\n=== VERIFIED LOCAL CLINIC SEARCH RESULTS (Location: {user_location}) ===\n"
                    f"{raw_search}\n"
                    f"=== END OF SEARCH RESULTS ===\n"
                    f"INSTRUCTIONS FOR DOCTORS: Extract up to 3 genuine clinic/doctor profiles from the search results above. "
                    f"Extract name, phone, locality, rating, and website links. If search results do not list real clinics, leave doctors empty."
                )

        # Build full clinical prompt
        full_input = (
            f"User Location: {user_location}\n"
            f"Target Specialty Needed: {intent.specialty}\n"
            f"{'Conversation History:\n' + history_text + '\n' if history_text else ''}"
            f"Current Patient Query: {user_prompt}"
            f"{doctor_context}"
        )

        # Tier 3: Main Clinical Triage Agent
        try:
            result = self._retry_run(self.health_agent, full_input)
            return result.output.model_dump()
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate limit" in err_msg.lower():
                return HealthResponse(
                    is_valid_query=True,
                    query_type="general_health",
                    direct_answer="I am currently experiencing high demand and reached the Groq API rate limit.",
                    advice="Please wait 30 seconds and try your request again. For urgent medical emergencies, immediately contact your local emergency hospital.",
                ).model_dump()
            raise e
