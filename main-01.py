from flair.data import Sentence
from flair.models import SequenceTagger

# Charger le modèle de NER de Flair
tagger = SequenceTagger.load("flair/ner-french")


def anonymize_with_flair(text):
    # Créer un objet Sentence pour le texte d'entrée
    sentence = Sentence(text)

    # Analyser les entités nommées avec Flair
    tagger.predict(sentence)

    anonymized_text = text
    for entity in sentence.get_spans('ner'):
        if entity.tag == "PER":
            anonymized_text = anonymized_text.replace(entity.text, "<PERSONNE>")
        elif entity.tag == "LOC":
            anonymized_text = anonymized_text.replace(entity.text, "<LIEU>")
        elif entity.tag == "ORG":
            anonymized_text = anonymized_text.replace(entity.text, "<ORGANISATION>")
        elif entity.tag == "MISC":
            anonymized_text = anonymized_text.replace(entity.text, "<DIVERS>")

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_with_flair(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
