from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

numero_de_dias = 15
numero_de_pessoas = 4
atividade = "praia"



prompt = f"Planeje uma viagem de {numero_de_dias} dias para {numero_de_pessoas} pessoas com foco em atividades de {atividade}. Forneça um itinerário detalhado, incluindo sugestões de hospedagem, restaurantes e atrações turísticas."
modelo = ChatAnthropic(
    model="claude-sonnet-5",
    anthropic_api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)