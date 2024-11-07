from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Charger le modèle BERT pour la reconnaissance d'entités nommées en français
model_name = "dbmdz/bert-base-french-europeana-cased"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Créer un pipeline pour la reconnaissance d'entités nommées (NER)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def anonymize_text_with_bert(text):
    # Utiliser BERT pour détecter les entités nommées
    entities = nlp_ner(text)
    anonymized_text = text
    for entity in entities:
        # Utiliser des noms significatifs pour chaque entité détectée
        if entity['entity_group'] == "PER":
            anonymized_text = anonymized_text.replace(entity['word'], "<PERSONNE>")
        elif entity['entity_group'] == "LOC":
            anonymized_text = anonymized_text.replace(entity['word'], "<LIEU>")
        elif entity['entity_group'] == "ORG":
            anonymized_text = anonymized_text.replace(entity['word'], "<ORGANISATION>")
        elif entity['entity_group'] == "MISC":
            anonymized_text = anonymized_text.replace(entity['word'], "<DIVERS>")

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_text_with_bert(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
