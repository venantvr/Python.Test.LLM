import re

from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle de NER de Flair
tagger = SequenceTagger.load("flair/ner-french")

# Charger le modèle GPT-2 pour la reformulation
model_name = "dbddv01/gpt2-french-small"  # GPT-2 en français
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def detect_and_replace_entities(text):
    # Détection des entités
    sentence = Sentence(text)
    tagger.predict(sentence)
    anonymized_text = text
    for entity in sentence.get_spans('ner'):
        if entity.tag == "PER":
            anonymized_text = anonymized_text.replace(entity.text, "Personne")
        elif entity.tag == "LOC":
            anonymized_text = anonymized_text.replace(entity.text, "Lieu")
        elif entity.tag == "ORG":
            anonymized_text = anonymized_text.replace(entity.text, "Organisation")
        elif entity.tag == "MISC" or entity.tag == "DATE":
            anonymized_text = anonymized_text.replace(entity.text, "Date")

    # Masquer les numéros de téléphone avec des expressions régulières
    anonymized_text = re.sub(r'\b\d{10}\b', 'NuméroDeTéléphone', anonymized_text)
    anonymized_text = re.sub(r'\b\d{2} \d{2} \d{2} \d{2} \d{2}\b', 'NuméroDeTéléphone', anonymized_text)
    anonymized_text = re.sub(r'\+33\s?\d{9}', 'NuméroDeTéléphone', anonymized_text)

    return anonymized_text


def rewrite_with_gpt2(text):
    # Préparer le texte pour la reformulation
    inputs = tokenizer.encode("Réécrire le texte pour anonymiser: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True)

    # Décoder le texte généré
    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten_text


def chatbot():
    print("Bienvenue dans l'agent d'anonymisation. Tapez 'exit' pour quitter.")
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == "exit":
            print("Agent : Au revoir!")
            break

        # Anonymiser et reformuler le texte
        anonymized_text = detect_and_replace_entities(user_input)
        rewritten_text = rewrite_with_gpt2(anonymized_text)

        print("Agent (texte anonymisé) :", rewritten_text)


# Lancer le chatbot
chatbot()
