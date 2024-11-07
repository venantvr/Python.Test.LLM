from sentence_transformers import SentenceTransformer

# Charger le modèle
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Liste de phrases à encoder
phrases = [
    "Ceci est une phrase d'exemple.",
    "Chaque phrase est convertie en un vecteur dense."
]

# Générer les embeddings
embeddings = model.encode(phrases)

# Afficher les embeddings
for i, embedding in enumerate(embeddings):
    print(f"Embedding pour la phrase {i + 1}: {embedding[:5]}...")  # Affiche les 5 premières valeurs de l'embedding
