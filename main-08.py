from sentence_transformers import SentenceTransformer, util

# Charger un modèle Sentence-BERT pour les textes biomédicaux
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Décrire les symptômes
symptome1 = "Le patient présente une toux sèche persistante accompagnée de fièvre modérée."
symptome2 = "Le patient a une engine."

# Obtenir les embeddings de phrases
embedding1 = model.encode(symptome1, convert_to_tensor=True)
embedding2 = model.encode(symptome2, convert_to_tensor=True)

# Calculer la similarité cosinus
similarity_score = util.cos_sim(embedding1, embedding2).item()

# Afficher le score de similarité
print(f"Score de similarité entre les descriptions de symptômes : {similarity_score:.4f}")
