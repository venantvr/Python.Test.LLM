from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Charger le modèle Camembert pour la reconnaissance d'entités nommées (NER)
model_name = "Jean-Baptiste/camembert-ner"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Forcer l'utilisation du tokenizer slow

# Créer un pipeline pour la reconnaissance d'entités nommées (NER)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def anonymize_text_with_camembert(text):
    # Utiliser Camembert pour détecter les entités nommées
    entities = nlp_ner(text)
    anonymized_text = text
    for entity in entities:
        # Remplacer chaque entité détectée par son type
        anonymized_text = anonymized_text.replace(entity['word'], f"<{entity['entity_group']}>")

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_text_with_camembert(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
