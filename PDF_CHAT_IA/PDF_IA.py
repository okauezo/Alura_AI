from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class PDFChatIA:
    def __init__(self):
        self.file = None
        self.documents = None

    def carregar_pdf(self):
        if self.file is None:
            raise ValueError("Nenhum arquivo PDF foi carregado.")

        loader = PyPDFLoader(self.file)
        self.documents = loader.load()

    def dividir_texto(self):
        if not hasattr(self, 'documents'):
            raise ValueError("Nenhum documento foi carregado para dividir.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(self.documents)

        self.embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


pdf = PDFChatIA()
pdf.file = "Regras_do_jogo_2023_24.pdf"
pdf.carregar_pdf()
pdf.dividir_texto()