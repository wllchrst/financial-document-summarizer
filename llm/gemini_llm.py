from llm.base_llm import BaseLLM
from google import genai
from google.genai import types
from helpers import env_helper


class GeminiLLM(BaseLLM):
    def __init__(self, system_prompt: str = None):
        super().__init__()
        self.API_KEY = env_helper.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.API_KEY)

    def answer(self, prompt: str, system_prompt: str, with_logging: bool):
        model = "gemini-2.0-flash-lite"

        contents = [
            types.Content(
                role="system",
                parts=[
                    types.Part.from_text(text=system_prompt),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
        )

        result = ""

        for chunk in self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
        ):
            if hasattr(chunk, "text") and chunk.text:
                result += chunk.text

        return result
