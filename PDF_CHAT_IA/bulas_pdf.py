import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from sentence_transformers import CrossEncoder

load_dotenv()

class bulas_pdf:
    def __init__(self):
        self.file = None
        self.documents = None
        self.chunks = None
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.reranker = None


    def carregar_pdf(self):
        if self.file is None:
            raise ValueError("Nenhum arquivo PDF foi carregado.")

        loader = PyPDFLoader(self.file)
        self.documents = loader.load()

    def dividir_em_chunks(self, chunks_size=800, chunks_overlap=150):
        if self.documents is None:
            raise ValueError("Nenhum documento foi carregado ainda. Rode carregar_pdf() primeiro.")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunks_size,
            chunk_overlap=chunks_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]

        )

# split_documents já preserva o metadata original de cada página (ex: page number)
        self.chunks = splitter.split_documents(self.documents)
        return self.chunks

    def enriquecer_metadados(self, categoria = "bula_farmaceutica"):
        if self.chunks is None:
            raise ValueError("Nenhum chunk foi gerado ainda. Rode dividir_em_chunks() primeiro.")
        
# Palavras-chave que ajudam a identificar em qual seção da bula o chunk está
        secoes_conhecidas = {
            "indicações": "Indicações",
            "contraindicações": "Contraindicações",
            "posologia": "Posologia",
            "reações adversas": "Reações Adversas",
            "modo de usar": "Modo de Uso"
        }

        for chunk in self.chunks:
            texto_minusculo = chunk.page_content.lower()

            secao_identificada = "Não identificada"

            for palavra_chave, nome_secao in secoes_conhecidas.items():
                if palavra_chave in texto_minusculo:
                    secao_identificada = nome_secao
                    break

        
            chunk.metadata["origem"] = self.file
            chunk.metadata["secao"] = secao_identificada
            chunk.metadata["categoria"] = categoria

        return self.chunks
    
    def gerar_embeddings(self, modelo="sentence-transformers/all-MiniLM-L6-v2"):
        if self.chunks is None:
            raise ValueError("Nenhum chunk foi gerado ainda. Rode dividir_em_chunks() primeiro.")

        self.embedding_model = HuggingFaceEmbeddings(model_name=modelo)
        textos = [chunk.page_content for chunk in self.chunks]
        vetores = self.embedding_model.embed_documents(textos)
        
        return vetores
    
    def criar_vector_store(self,persist_directory = "chroma_db"):
        if self.chunks is None:
            raise ValueError("Nenhum chunk foi gerado ainda. Rode dividir_em_chunks() primeiro.")
        if self.embedding_model is None:
            raise ValueError("Nenhum embedding foi gerado ainda. Rode gerar_embeddings() primeiro.")


        self.vector_store = Chroma.from_documents(
            documents = self.chunks,
            embedding = self.embedding_model,
            persist_directory = persist_directory
        )

        return self.vector_store
    
    def configurar_llm(self, modelo="claude-sonnet-5", api_key=None):

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(model=modelo, api_key=api_key)

    def perguntar(self, pergunta, k_inicial=10, k_final=3):
        if self.vector_store is None:
            raise ValueError("Nenhum vector store foi criado ainda. Rode criar_vector_store() primeiro.")
        if self.llm is None:
            raise ValueError("Nenhum LLM foi configurado ainda. Rode configurar_llm() primeiro.")
        if self.reranker is None:
            raise ValueError("Nenhum reranker foi configurado ainda. Rode configurar_reranker() primeiro.")


        chunks_candidatos = self.vector_store.similarity_search(pergunta, k=k_inicial)
        pares = [[pergunta, chunk.page_content] for chunk in chunks_candidatos]
        notas = self.reranker.predict(pares)

        chunks_com_notas = list(zip(chunks_candidatos, notas))
        chunks_com_notas.sort(key=lambda item: item[1], reverse=True)


        chunks_relevantes = [chunk for chunk, nota in chunks_com_notas[:k_final]]
        contexto = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        
        prompt = f"""Responda a pergunta abaixo usando APENAS as informações do contexto fornecido. Se a resposta não estiver no contexto, diga que não sabe.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta: """

        resposta = self.llm.invoke(prompt)

        return {
            "resposta": resposta.content,
            "trechos_usados": chunks_relevantes
        }
    


if __name__ == "__main__":
    minha_bula = bulas_pdf()
    minha_bula.file = "dipirona.pdf"

    minha_bula.carregar_pdf()
    minha_bula.dividir_em_chunks()
    minha_bula.enriquecer_metadados()
    minha_bula.gerar_embeddings()
    minha_bula.criar_vector_store()
    minha_bula.configurar_llm()

    resultado = minha_bula.perguntar("Posso tomar esse remédio na gravidez?")
    print(resultado["resposta"])

    for trecho in resultado["trechos_usados"]:
        print("---")
        print(trecho.page_content[:200])
        print(trecho.metadata)