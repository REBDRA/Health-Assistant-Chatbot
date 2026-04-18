import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# 1. Define the exact structure for the Doctors
class Doctor(BaseModel):
    name: str = Field(
        description="Must include specialty in brackets, e.g. 'Dr. Amit Sharma (Ophthalmologist)'. Generate realistic Indian names."
    )
    phone: str
    location: str = Field(description="Must be a location in West Bengal, India.")
    rating: str = Field(description="Format: 'X.X/5'")


# 2. Define the main response structure
class HealthResponse(BaseModel):
    is_valid_query: bool = Field(
        description="False if user states gibberish, contradiction, or non-health joke."
    )
    query_type: str = Field(
        description="Must be exactly 'symptom_triage' or 'general_health'"
    )
    direct_answer: str = Field(
        default="", description="Fill ONLY if query_type is 'general_health'."
    )
    remedies: list[str] = Field(
        default=[], description="Home remedies or solutions. ONLY for 'symptom_triage'."
    )
    advice: str = Field(default="", description="General health advice.")
    doctors: list[Doctor] = Field(
        default=[],
        description="Exactly 3 recommended doctors. ONLY for 'symptom_triage'.",
    )
    error_message: str = Field(
        default="", description="Polite error message if is_valid_query is false."
    )


# 3. Initialize the PydanticAI Agent
# Llama 3.3 70B is highly reliable for strict schema generation on Groq
health_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    result_type=HealthResponse,
    system_prompt=(
        "You are a highly intelligent and strict Medical Triage & Health AI. "
        "Evaluate the user's input. If it is a symptom, provide triage, remedies, and 3 realistic doctors. "
        "If it is a general health question, provide a direct answer and advice. "
        "If it is not a health query, reject it politely."
    ),
)


# 4. The Facade to connect to your Streamlit app
class HealthAIFacade:
    def __init__(self, api_key: str):
        # PydanticAI automatically looks for the GROQ_API_KEY environment variable
        os.environ["GROQ_API_KEY"] = api_key

    def get_structured_response(self, user_prompt: str, chat_history: list) -> dict:
        # Format a brief chat history to give the AI context without overwhelming it
        history_text = ""
        if chat_history:
            recent_history = [msg for msg in chat_history if not msg.get("is_card")][
                -3:
            ]
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        full_prompt = (
            f"Chat Context:\n{history_text}\n\nUser's latest message: {user_prompt}"
        )

        # Run the agent. If the LLM breaks the schema, PydanticAI automatically retries!
        result = health_agent.run_sync(full_prompt)

        # Convert the resulting Pydantic object back to a dictionary for Streamlit
        return result.data.model_dump()
