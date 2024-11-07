from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Charger distilBERT pour la reconnaissance d'entités nommées (NER)
model_name = "distilbert-base-cased"
model = AutoModelForTokenClassification.from_pretrained("dbmdz/distilbert-base-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-cased-finetuned-conll03-english")

# Créer un pipeline pour la reconnaissance d'entités nommées (NER)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def anonymize_text_with_distilbert(text):
    # Utiliser distilBERT pour détecter les entités nommées
    entities = nlp_ner(text)
    anonymized_text = text
    for entity in entities:
        # Remplacer chaque entité détectée par son type (par exemple, PERSON, LOCATION)
        anonymized_text = anonymized_text.replace(entity['word'], f"<{entity['entity_group']}>")

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_text_with_distilbert(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
