from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

textos = [
    "O paciente pode consultar seus exames pelo portal.",
    "O agendamento deve ser feito pelo aplicativo.",
    "Em caso de dúvidas, entre em contato com o atendimento."
]

vetores = modelo.encode(textos)
for texto, vetor in zip(textos, vetores):
    print("Texto:", texto)
    print("Vetor:", vetor[:10])
    print("Tamanho do vetor:", len(vetor))
    print("-" * 50)