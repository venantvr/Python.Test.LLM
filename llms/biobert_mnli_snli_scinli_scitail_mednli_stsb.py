from sentence_transformers import SentenceTransformer, util


def biobert_mnli_snli_scinli_scitail_mednli_stsb(symptome1, symptome2):
    # Charger un modèle Sentence-BERT pour les textes biomédicaux
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    # Obtenir les embeddings de phrases
    embedding1 = model.encode(symptome1, convert_to_tensor=True)
    embedding2 = model.encode(symptome2, convert_to_tensor=True)
    # Calculer la similarité cosinus
    similarity_score = util.cos_sim(embedding1, embedding2).item()
    return similarity_score


if __name__ == "__main__":
    # Décrire les symptômes
    symptome1 = "Le patient présente une toux sèche persistante accompagnée de fièvre modérée."
    symptome2 = "Le patient a une engine."

    similarity_score = biobert_mnli_snli_scinli_scitail_mednli_stsb(symptome1, symptome2)
    # Afficher le score de similarité
    print(f"Score de similarité entre les descriptions de symptômes : {similarity_score:.4f}")
