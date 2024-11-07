import json

from transformers import pipeline

# Charger le modèle CamemBERT pour la reconnaissance d'entités nommées
model_name = "Jean-Baptiste/camembert-ner"
ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

# Définir des mots-clés pour différentes catégories médicales
symptoms_keywords = ["toux", "fièvre", "douleur", "maux", "sensation", "nausée", "vomissement", "fatigue"]
treatment_keywords = ["traitement", "médicament", "repos", "chirurgie", "paracétamol", "antibiotique", "prescription"]
diagnosis_keywords = ["diagnostic", "maladie", "syndrome", "inflammation", "infection"]


def classify_entity(entity_text):
    # Déterminer la catégorie en fonction des mots-clés
    if any(keyword in entity_text.lower() for keyword in symptoms_keywords):
        return "symptome"
    elif any(keyword in entity_text.lower() for keyword in treatment_keywords):
        return "traitement"
    elif any(keyword in entity_text.lower() for keyword in diagnosis_keywords):
        return "diagnostic"
    return "autre"


def text_to_json(text: str) -> str:
    # Utiliser CamemBERT pour détecter les entités dans le texte
    entities = ner_pipeline(text)

    # Initialiser la structure JSON
    structured_data = {
        "symptomes": [],
        "traitements": [],
        "diagnostics": [],
        "autres": []
    }

    # Classifier chaque entité détectée
    for entity in entities:
        category = classify_entity(entity["word"])
        entity_data = {
            "text": entity["word"],
            "type": entity["entity_group"],
            "start": entity["start"],
            "end": entity["end"]
        }

        # Ajouter l'entité à la bonne catégorie
        if category == "symptome":
            structured_data["symptomes"].append(entity_data)
        elif category == "traitement":
            structured_data["traitements"].append(entity_data)
        elif category == "diagnostic":
            structured_data["diagnostics"].append(entity_data)
        else:
            structured_data["autres"].append(entity_data)

    # Convertir le dictionnaire en JSON
    json_data = json.dumps(structured_data, indent=4, ensure_ascii=False)
    return json_data


# Exemple de texte médical
text = ("Le patient présente une toux sèche persistante accompagnée de fièvre modérée. On lui a prescrit "
        "du paracétamol pour soulager la douleur.")

# Convertir le texte en JSON structuré
json_output = text_to_json(text)
print(json_output)
