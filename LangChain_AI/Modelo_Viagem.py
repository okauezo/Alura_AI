from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

class Destino(BaseModel):
    cidade: str = Field(description="A Cidade sugerida para viagem")
    motivo: str = Field(description="O motivo para visitar a cidade")


class restaurtantes(BaseModel):
    cidade: str = Field(description="A Cidade sugerida para visitar")
    restaurantes: str = Field(description="Restaurantes recomendados da cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

promt_cidade = PromptTemplate(
    template=""" 
    Sugira uma cidade dado o meu interrese por {interesse}.
    {formato_de_saida}
    """,
    input_variables = ["interesse"],
    partial_variables = {"formato_de_saida": parseador_destino.get_format_instructions() }
)

prompt_restaurantes = PromptTemplate(
    template="""
Sugira restaurantes pouplares entre locais em {cidade}
{formato_de_saida}
""",
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

modelo = ChatAnthropic(
    model="claude-sonnet-5",
    anthropic_api_key=api_key
)

cadeia = promt_cidade | modelo | parseador_destino

resposta = cadeia.invoke(
    {
        "interesse": "montanhas"
    }
)
print(resposta)