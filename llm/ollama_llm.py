import httpx
from llm.base_llm import BaseLLM
from helpers import env_helper
from ollama import Client

timeout_seconds = 180


class OllamaLLM(BaseLLM):
    def __init__(self, llm_model=env_helper.OLLAMA_MODEL):
        super().__init__(llm_model)
        self.HOST = env_helper.OLLAMA_HOST
        self.client = Client(host=self.HOST, timeout=timeout_seconds)
        self.model_name = llm_model

    def answer(self,
               system_prompt: str,
               prompt: str,
               with_logging: bool) -> str:
        try:
            response = self.client.chat(self.llm_model, think=False, stream=False, keep_alive=-1, messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])

            answer = response.message.content

            if with_logging:
                print(f'System:\n{system_prompt}\n\n')
                print(f'User:\n{prompt}\n\n')
                print(f'Response Content:\n{answer}')

            return answer
        except httpx.ReadTimeout as timeout:
            raise TimeoutError(
                f'Ollama request exceeded timeout limit for model {self.model_name} with error: {timeout}')
        except Exception as e:
            raise RuntimeError(f"Error when processing in ollama: {str(e)}")
