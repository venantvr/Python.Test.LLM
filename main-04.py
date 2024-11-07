from sentence_transformers import SentenceTransformer, util

# Charger le modèle de paraphrase
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Définir deux phrases pour comparer
phrase1 = "Jean a le rhume."
phrase2 = "Marc a le nez qui coule."

# Générer les embeddings pour chaque phrase
embedding1 = model.encode(phrase1, convert_to_tensor=True)
embedding2 = model.encode(phrase2, convert_to_tensor=True)

# Calculer la similarité cosinus entre les embeddings
similarity_score = util.cos_sim(embedding1, embedding2).item()

# Afficher le score de similarité
print(f"Score de similarité entre les phrases : {similarity_score:.4f}")
