from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texto = "Estou aprendendo embeddings na Alura"

vetor = modelo.encode(texto)

print("Texto original:")
print(texto)

print("\nPrimeiros 5 números do vetor:")
print(vetor[:5])

print("\nTamanho do vetor:")
print(len(vetor))
