from langchain_anthropic import ChatAnthropic
from core.config import settings

def get_llm():
    """Initializes the LLM used by the supervisor and sub-agents."""
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0
    )