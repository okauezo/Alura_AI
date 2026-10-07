from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

class Destino(BaseModel):
    cidade: str = Field(..., description="Cidade sugerida para viagem")
    pais: str = Field(..., description="País da cidade sugerida")
    descricao: str = Field(..., description="Descrição da cidade e suas atrações turísticas")

promt_cidade = PromptTemplate(
    template=""" 
    Sugira uma cidade dado o meu interrese por {interesse}.
    """,
    input_variables = ["interesse"]
)

modelo = ChatAnthropic(
    model="claude-sonnet-5",
    anthropic_api_key=api_key
)

cadeia = promt_cidade | modelo | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse": "montanhas"
    }
)
print(resposta)