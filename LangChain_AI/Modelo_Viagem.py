from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

modelo_cidade = PromptTemplate(
    template=""" 
    Sugira uma cidade dado o meu interrese por {interesse}.
    """,
    input_variables = ["interesse"]
)


print("Prompt : \n", prompt)  
modelo = ChatAnthropic(
    model="claude-sonnet-5",
    anthropic_api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)