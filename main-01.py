from transformers import pipeline

# Utiliser le modèle Camembert standard pour le NER
nlp_ner = pipeline("ner", model="camembert-base", tokenizer="camembert-base", aggregation_strategy="simple")

# Mapping de labels pour des noms plus explicites (selon les besoins du modèle)
label_mapping = {
    "LABEL_0": "PERSONNE",
    "LABEL_1": "LIEU",
    "LABEL_2": "ORGANISATION",
    "LABEL_3": "DATE",
}


def anonymize_text_with_camembert(text):
    # Utiliser Camembert pour détecter les entités nommées
    entities = nlp_ner(text)
    anonymized_text = text
    for entity in entities:
        # Utiliser le mapping des labels pour remplacer chaque entité détectée
        label = label_mapping.get(entity['entity'], "ENTITE")
        anonymized_text = anonymized_text.replace(entity['word'], f"<{label}>")

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_text_with_camembert(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
