from sentence_transformers import SentenceTransformer, util


def paraphrase_multilingual_minilm(symptome1, symptome2):
    # Charger le modèle de paraphrase multilingue
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # Générer les embeddings pour chaque description de symptôme
    embedding1 = model.encode(symptome1, convert_to_tensor=True)
    embedding2 = model.encode(symptome2, convert_to_tensor=True)

    # Calculer la similarité cosinus entre les embeddings
    similarity_score = util.cos_sim(embedding1, embedding2).item()

    return similarity_score


if __name__ == "__main__":
    # Décrire deux symptômes médicaux pour comparaison
    symptome1 = "Le patient présente une toux sèche persistante accompagnée de fièvre modérée."
    # symptome2 = "Le patient a une fièvre légère et tousse de manière intermittente sans production de mucus."
    symptome2 = "Le patient s'est cassé le bras."

    similarity_score = paraphrase_multilingual_minilm(symptome1, symptome2)
    # Afficher le score de similarité
    print(f"Score de similarité entre les descriptions de symptômes : {similarity_score:.4f}")
