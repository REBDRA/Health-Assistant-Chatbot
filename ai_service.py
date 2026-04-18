# ai_service.py
import json


class HealthAIFacade:
    def __init__(self, client, system_prompt):
        self.client = client
        self.system_prompt = system_prompt

    def get_structured_response(self, user_prompt: str, chat_history: list) -> dict:
        # We build the full memory for the AI here
        api_messages = [{"role": "system", "content": self.system_prompt}]

        # Add the history so the AI remembers the conversation
        for msg in chat_history:
            # Only send user and assistant roles to the API
            if msg["role"] in ["user", "assistant"]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the brand new prompt
        api_messages.append({"role": "user", "content": user_prompt})

        # The actual API call is hidden inside this function
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)
