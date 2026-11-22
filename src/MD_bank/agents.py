from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.prompts import HumanMessage
from langchain.chat_models import init_chat_model
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

_llm = init_chat_model(
    model="claude-sonnet-5.5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=1000
)