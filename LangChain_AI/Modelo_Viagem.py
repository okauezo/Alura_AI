from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

class Destino(BaseModel):
    cidade: str = Field("A Cidade sugerida para viagem")
    motivo: str = Field("O motivo para visitar a cidade")

parseador = JsonOutputParser(pydantic_object=Destino)

promt_cidade = PromptTemplate(
    template=""" 
    Sugira uma cidade dado o meu interrese por {interesse}.
    """,
    input_variables = ["interesse"]
    partial_variables = {"format_instructions": parseador.get_format_instructions() }
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