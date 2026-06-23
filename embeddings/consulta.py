from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

