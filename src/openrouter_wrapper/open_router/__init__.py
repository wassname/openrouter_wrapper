from typing import Optional
import dotenv
import os
dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
