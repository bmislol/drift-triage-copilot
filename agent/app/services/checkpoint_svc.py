import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
import sys
from pathlib import Path

# Path hack for core imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from core.config import settings

def get_checkpointer():
    """
    Creates a persistent checkpointer instance with autocommit enabled.
    Autocommit is required for LangGraph to run its internal migrations
    (like creating concurrent indexes).
    """
    # Initialize the connection with autocommit=True
    conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
    return PostgresSaver(conn)