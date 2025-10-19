import os
from dotenv import load_dotenv, find_dotenv

ENVS = ["GEMINI_API_KEY", "OLLAMA_HOST", "OLLAMA_MODEL"]


class EnvHelper:
    """Class for gathering and saving all env for the application """

    def __init__(self):
        env_path = find_dotenv()
        print(f'Loading env from {env_path}')
        load_dotenv(env_path, override=True)
        self.envs = {}

        self.gather_envs()
        self.assign_env()

    def gather_envs(self) -> bool:
        """Gather All env for the application if there is a missing value throws error

        Returns:
            bool: _description_
        """
        for env in ENVS:
            env_value = os.getenv(env)
            if env_value is None:
                raise ValueError(f'{env} has value None')

            self.envs[env] = os.getenv(env)

        return True

    def assign_env(self):
        self.GEMINI_API_KEY = self.envs[ENVS[0]]
        self.OLLAMA_HOST = self.envs[ENVS[1]]
        self.OLLAMA_MODEL = self.envs[ENVS[2]]


env_helper = EnvHelper()
