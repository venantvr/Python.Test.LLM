import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


# Fonction pour obtenir les embeddings moyens d'une phrase
def get_sentence_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    # Calculer la moyenne des embeddings des tokens
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def camembert_base(symptome1, symptome2):
    # Charger le modèle CamemBERT standard
    model_name = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Obtenir les embeddings moyens pour chaque description de symptôme
    embedding1 = get_sentence_embedding(symptome1)
    embedding2 = get_sentence_embedding(symptome2)
    # Calculer la similarité cosinus entre les embeddings
    similarity_score = cosine_similarity(embedding1.numpy(), embedding2.numpy())[0][0]
    return similarity_score


if __name__ == "__main__":
    # Décrire deux symptômes médicaux pour comparaison
    symptome1 = "Le patient présente une toux sèche persistante accompagnée de fièvre modérée."
    # symptome2 = "Le patient a une fièvre légère et tousse de manière intermittente sans production de mucus."
    symptome2 = "Le patient s'est cassé le bras."

    similarity_score = camembert_base(symptome1, symptome2)
    # Afficher le score de similarité
    print(f"Score de similarité entre les descriptions de symptômes : {similarity_score:.4f}")
