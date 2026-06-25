from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documentos = [

    "Como criar uma AI para auxiliar no atendimento ao cliente?",
    "Quais são as melhores práticas para implementar uma IA no atendimento ao cliente?",
    "Como medir o impacto da IA no atendimento ao cliente?"
]

vetores_documentos = modelo.encode(documentos)

consulta = "Quais são os benefícios de usar IA no atendimento ao cliente?"

vetor_consulta = modelo.encode([consulta])

similaridades = cosine_similarity(vetor_consulta, vetores_documentos)

indice_mais_parecido = similaridades[0].argmax()



