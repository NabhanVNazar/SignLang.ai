
import os
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client

ROOT_ENV = os.path.abspath(".env")
load_dotenv(ROOT_ENV)

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
        )

    return create_client(url, key)
