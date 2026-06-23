from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documentos = [

    "Como criar uma AI para auxiliar no atendimento ao cliente?",
    "Quais são as melhores práticas para implementar uma IA no atendimento ao cliente?",
    "Como medir o impacto da IA no atendimento ao cliente?"
]



