import chromadb
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

textos = [
    "O paciente pode consultar seus exames pelo portal.",
    "O agendamento deve ser feito pelo aplicativo.",
    "Em caso de dúvidas, entre em contato com o atendimento."
]

vetores = modelo.encode(textos)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documentos_alura")

ids = ["doc1", "doc2", "doc3"]

metadados = [
    {"tipo": "exames"},
    {"tipo": "agendamento"},
    {"tipo": "atendimento"}
]

collection.upsert(
    ids=ids,
    documents=textos,
    embeddings=vetores.tolist(),
    metadatas=metadados
)   

pergunta = "Como posso agendar uma consulta?"

vetor_pergunta = modelo.encode([pergunta])
resultados = collection.query(
    query_embeddings=vetor_pergunta.tolist(),
    n_results=2
)

