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
        description="Name of the specialist or clinic (e.g., 'Dr. Amitav Banerjee' or 'BM Birla Heart Research Centre')."
    )
    specialty: str = Field(
        default="",
        description="Exact medical specialty or subspecialty (e.g., 'Cardiologist', 'Ophthalmologist')."
    )
    clinic_or_hospital: str = Field(
        default="",
        description="Associated hospital, medical center, or clinic facility name."
    )
    location: str = Field(
        default="Local area",
        description="Specific street address, neighborhood, or locality near the user."
    )
    phone: str = Field(
        default="Visit Website",
        description="Direct appointment phone number or booking hotline."
    )
    rating: str = Field(
        default="4.6/5",
        description="Rating from search results (format X.X/5) or 'Verified'."
    )
    link: str = Field(
        default="",
        description="Direct official website URL or booking profile link."
    )
    why_recommended: str = Field(
        default="",
        description="1 concise sentence explaining why this specialist specifically matches the patient's condition."
    )


class HealthResponse(BaseModel):
    is_valid_query: bool = Field(
        description="True if query is health/medical related, False for off-topic/nonsense."
    )
    query_type: str = Field(
        description="'symptom_triage' for physical/mental complaints, 'general_health' for educational health questions."
    )
    urgency_level: str = Field(
        default="Routine",
        description="'Emergency (Call 112/911)', 'Urgent (Consult within 24-48h)', or 'Routine / Self-Care'."
    )
    specialty_needed: str = Field(
        default="",
        description="Primary medical specialty needed (e.g. 'Neurologist', 'Cardiologist', 'Dermatologist')."
    )
    direct_answer: str = Field(
        default="",
        description="Comprehensive clinical evaluation or symptom overview."
    )
    remedies: List[str] = Field(
        default_factory=list,
        description="Actionable, safe home remedies, recovery steps, or first-aid measures."
    )
    advice: str = Field(
        default="",
        description="Clear medical guidance, preventive measures, and red-flag symptoms to monitor."
    )
    doctors: List[Doctor] = Field(
        default_factory=list,
        description="Top matching local doctors or specialized clinics extracted strictly from search results."
    )
    error_message: str = Field(
        default="",
        description="Polite message explaining why query could not be processed if invalid."
    )


class QueryIntent(BaseModel):
    is_health_related: bool = Field(
        description="True if query pertains to health, medicine, fitness, symptoms, or biology."
    )
    is_medically_coherent: bool = Field(
        default=True,
        description="False if the query contains anatomically contradictory, nonsensical, or physically impossible symptom combinations (e.g., 'headache in knee', 'toothache in foot', 'broken stomach bone'). True if it describes a plausible condition, inquiry, or symptom."
    )
    is_symptom: bool = Field(
        description="True if user describes an active physical or mental complaint requiring triage."
    )
    specialty: str = Field(
        default="General Physician",
        description="The precise medical specialist needed (e.g. Cardiologist, Dermatologist, ENT Specialist, Neurologist)."
    )
    search_keywords: str = Field(
        default="",
        description="2 to 4 concise keywords describing the medical condition (e.g. 'migraine headache neurology')."
    )
    clarification_message: str = Field(
        default="",
        description="If not medically coherent, a polite, professional clinical explanation asking the patient to clarify their symptoms without assuming or fabricating."
    )


# --- 2. SYSTEM INSTRUCTIONS ---

HEALTH_SYSTEM_PROMPT = """You are an elite, certified Medical Triage AI and Clinical Doctor Discovery Engine.
Your primary mission is to help patients understand their symptoms accurately, provide immediate safe relief steps, and precisely connect them to verified local medical specialists suited for their exact health condition.

CRITICAL INSTRUCTIONS FOR SPECIALIST & CLINIC MATCHING (CORE PURPOSE):
1. Precision Matching:
   - Identify the exact medical discipline and sub-specialty required for the patient's complaint (e.g., 'Interventional Cardiologist', 'Corneal Eye Specialist', 'Orthopedic Spine Surgeon', 'ENT Specialist').
   - Set 'specialty_needed' clearly.
   - Extract up to 3 genuine doctors, hospitals, or specialized clinics from the VERIFIED LOCAL SEARCH RESULTS provided below.
   - Fill in:
     - name: Doctor or Clinic name.
     - specialty: Exact specialty matching the symptom.
     - clinic_or_hospital: Hospital/Center affiliation.
     - location: Specific address/locality.
     - phone: Contact number or appointment booking hotline.
     - rating: 'X.X/5' or 'Verified'.
     - link: URL to profile or clinic website.
     - why_recommended: 1 concise sentence explaining specifically why this specialist is the right choice for the user's symptoms.
2. Accuracy & Medical Integrity:
   - If a query contains anatomically contradictory or nonsensical combinations (e.g., 'headache in knee', 'ear pain in elbow', 'toothache in foot'):
     DO NOT invent excuses or assume metaphors (e.g. NEVER say 'sometimes described as a headache in the knee').
     Set is_valid_query=false, and provide a polite, professional clarification explaining that a headache is cranial pain while knee pain is an orthopedic joint issue, and invite them to specify their exact symptoms.
   - Only extract real clinics, doctors, and contact numbers found in the provided search results. Never invent fake phone numbers or addresses.
   - If no doctors are in search results, leave the doctors list empty.
3. Urgency:
   - Set urgency_level: 'Emergency (Call 112/911)', 'Urgent (Consult within 24-48h)', or 'Routine / Self-Care'.
   - If symptoms indicate life-threatening conditions (e.g. crushing chest pain, sudden numbness/slurred speech, severe breathing distress), prominently urge emergency services immediately.
4. Remedies & Advice:
   - Keep remedies concise, bulleted, and medically safe (e.g. hydration, rest, specific over-the-counter or non-drug remedies).
   - Keep advice clear, actionable, and highlight red-flag symptoms.
"""

INTENT_SYSTEM_PROMPT = """You are an ultra-fast, rigorous clinical query analyzer.
Analyze the user query with strict medical prudence and anatomical coherence:
1. is_health_related: boolean (false for finance, coding, politics, or general trivia)
2. is_medically_coherent: boolean
   - Set to FALSE if the query contains anatomically contradictory, nonsensical, or physically impossible phrases (e.g., 'headache in knee', 'heart attack in finger', 'coughing through my ears', 'stomach bone fracture').
   - Do NOT try to stretch or rationalize nonsensical metaphors (e.g., do NOT assume 'headache in knee' means knee pain).
   - If FALSE, write a professional, polite clarification_message explaining why the phrase is medically contradictory and asking them to clarify which specific part of their body is affected.
3. is_symptom: boolean (true if user describes an active, coherent bodily or mental symptom requiring triage)
4. specialty: the appropriate medical specialist if coherent
5. search_keywords: 2 to 4 keywords for finding clinics
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
    """Fetches verified clinics and doctor listings using targeted multi-query search."""
    clean_loc = sanitize_search_term(location)
    clean_spec = sanitize_search_term(specialty)
    clean_kw = sanitize_search_term(keywords) if keywords else ""

    # Primary targeted query for hospital/clinic with contact details
    q1 = f"best {clean_spec} doctor clinic in {clean_loc} hospital phone address"
    q1 = " ".join(q1.split())
    raw_results = _cached_doctor_search(q1)

    # If results are sparse, supplement with top specialist query
    if not raw_results or len(raw_results) < 250:
        q2 = f"top {clean_spec} specialist in {clean_loc} contact appointment"
        q2 = " ".join(q2.split())
        supp = _cached_doctor_search(q2)
        if supp:
            raw_results = (raw_results + "\n\n---\n\n" + supp).strip()

    return raw_results


os.environ["PYDANTIC_AI_NO_BANNER"] = "1"


def resolve_groq_models(api_key: str) -> Tuple[str, str]:
    """Detects best available models for fast classification and main triage."""
    from groq import Groq
    try:
        client = Groq(api_key=api_key)
        available = {m.id for m in client.models.list().data}

        # Fast model candidates
        fast_candidates = [
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
            "qwen/qwen3.8-27b",
            "groq/compound-mini",
        ]
        fast_model = next((m for m in fast_candidates if m in available), "openai/gpt-oss-20b")

        # Triage model candidates
        triage_candidates = [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
        ]
        triage_model = next((m for m in triage_candidates if m in available), "openai/gpt-oss-120b")

        return fast_model, triage_model
    except Exception:
        return "openai/gpt-oss-20b", "openai/gpt-oss-120b"


# --- 4. FACADE & ORCHESTRATION ---

class HealthAIFacade:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key cannot be empty.")

        self.api_key = api_key.strip()
        os.environ["GROQ_API_KEY"] = self.api_key

        self.provider = GroqProvider(api_key=self.api_key)

        fast_model_id, triage_model_id = resolve_groq_models(self.api_key)

        # Fast tier: sub-300ms intent & specialty classification
        self.fast_model = GroqModel(fast_model_id, provider=self.provider)
        self.intent_agent = Agent(
            model=self.fast_model,
            output_type=QueryIntent,
            system_prompt=INTENT_SYSTEM_PROMPT,
        )

        # Clinical tier: high-capacity reasoning & structured triage
        self.triage_model = GroqModel(triage_model_id, provider=self.provider)
        self.health_agent = Agent(
            model=self.triage_model,
            output_type=HealthResponse,
            system_prompt=HEALTH_SYSTEM_PROMPT,
        )

    def validate_key(self) -> Tuple[bool, str]:
        """Validates the Groq API key with a minimal request."""
        try:
            test_agent = Agent(self.fast_model, output_type=str)
            test_agent.run_sync("ping")
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

        # Reject anatomically contradictory or nonsensical queries
        if not intent.is_medically_coherent:
            msg = intent.clarification_message or (
                "Your query appears to combine anatomically contradictory terms. "
                "For example, a headache refers specifically to cranial pain, whereas joint discomfort affects areas like the knee. "
                "Please clarify which specific symptom and body area you are experiencing so I can assist you properly."
            )
            return HealthResponse(
                is_valid_query=False,
                query_type="invalid",
                error_message=msg,
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
                    f"\n\n=== VERIFIED LOCAL SPECIALIST & CLINIC SEARCH RESULTS (Location: {user_location}) ===\n"
                    f"{raw_search}\n"
                    f"=== END OF SEARCH RESULTS ===\n"
                    f"INSTRUCTIONS FOR DOCTORS: Extract up to 3 genuine clinic/doctor profiles from the search results above. "
                    f"Extract name, specialty, clinic/hospital name, specific location/address, phone number, rating, link, and why_recommended. "
                    f"If search results do not list real clinics, leave doctors empty."
                )

        # Build full clinical prompt
        full_input = (
            f"User Location: {user_location}\n"
            f"Target Specialist Discipline: {intent.specialty}\n"
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
