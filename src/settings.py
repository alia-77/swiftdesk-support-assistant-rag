import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

CHROMA_PERSIST_DIRECTORY = "chroma_db"

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_RETRIEVAL_COUNT = 3

