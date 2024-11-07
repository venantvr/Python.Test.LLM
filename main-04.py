import re

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Charger le modèle NER de CamemBERT pour détecter les entités en français
ner_model = "Jean-Baptiste/camembert-ner"
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_model, aggregation_strategy="simple")

# Charger le modèle de paraphrase multilingue
paraphrase_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def detect_and_replace_entities(text):
    # Détection des entités nommées avec CamemBERT
    entities = ner_pipeline(text)
    anonymized_text = text
    for entity in entities:
        if entity['entity_group'] == "PER":
            anonymized_text = anonymized_text.replace(entity['word'], "Personne")
        elif entity['entity_group'] == "LOC":
            anonymized_text = anonymized_text.replace(entity['word'], "Lieu")
        elif entity['entity_group'] == "ORG":
            anonymized_text = anonymized_text.replace(entity['word'], "Organisation")
        elif entity['entity_group'] == "MISC" or entity['entity_group'] == "DATE":
            anonymized_text = anonymized_text.replace(entity['word'], "Date")

    # Masquer les numéros de téléphone avec des expressions régulières
    anonymized_text = re.sub(r'\b\d{10}\b', 'NuméroDeTéléphone', anonymized_text)
    anonymized_text = re.sub(r'\b\d{2} \d{2} \d{2} \d{2} \d{2}\b', 'NuméroDeTéléphone', anonymized_text)
    anonymized_text = re.sub(r'\+33\s?\d{9}', 'NuméroDeTéléphone', anonymized_text)

    return anonymized_text


def paraphrase_text(text):
    # Générer des paraphrases en utilisant le modèle de paraphrase multilingue
    paraphrased_embeddings = paraphrase_model.encode([text], convert_to_tensor=True)
    paraphrase_candidates = util.paraphrase_mining(paraphrase_model, [text])[0][2]

    # Retourner la paraphrase sans ajout de contenu
    return paraphrase_candidates


def chatbot():
    print("Bienvenue dans l'agent d'anonymisation. Tapez 'exit' pour quitter.")
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == "exit":
            print("Agent : Au revoir!")
            break

        # Anonymiser les entités détectées dans le texte
        anonymized_text = detect_and_replace_entities(user_input)

        # Paraphraser le texte anonymisé sans inventer de contexte
        rewritten_text = paraphrase_text(anonymized_text)

        print("Agent (texte anonymisé) :", rewritten_text)


# Lancer le chatbot
chatbot()
