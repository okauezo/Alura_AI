from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

numero_de_dias = 15
numero_de_pessoas = 4
atividade = "praia"

modelo_de_prompt = PromptTemplate(
    template=""" 
    Crie um roteiro de viajem de {dias} dias, 
    para uma familia com {numero_de_pessoas} crinças,
    que gostamd de {atividade}.
    """
)

prompt = modelo_de_prompt.format(
    dias=numero_de_dias,
    numero_de_pessoas=numero_de_pessoas,
    atividade=atividade
)

print("Prompt : \n", prompt)  
modelo = ChatAnthropic(
    model="claude-sonnet-5",
    anthropic_api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)