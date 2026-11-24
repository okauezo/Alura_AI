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

agente_cartao_credito = create_agent(
    _llm,
    tools=[],
    system_prompt=("Você é um assistente virtual especializado em ajudar clientes com informações sobre seus cartões de crédito do Banco MDBank."
                   "Ajuda o cliente com duvidas sobre limites, faturas, transações e benefícios do cartão de crédito."),
)

agente_abertura_conta = create_agent(
    _llm,
    tools=[],
    system_prompt=(
        "Você é um especialista em abertura de contas do banco MDBank. "
        "Ajude o cliente a abrir uma conta e explique os tipos disponíveis."
    ),
)