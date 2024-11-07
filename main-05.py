from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Charger le modèle BioBERT pour l'extraction d'entités
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer un pipeline pour la tâche de NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

import json


def extract_medical_entities(text: str) -> dict:
    # Extraire les entités nommées
    entities = ner_pipeline(text)

    # Initialiser des listes pour chaque catégorie
    structured_data = {"symptomes": [], "traitements": [], "diagnostics": []}

    # Classifier les entités extraites en catégories
    for entity in entities:
        entity_text = entity['word']
        entity_label = entity['entity']

        # Classification simplifiée (à adapter selon les besoins)
        if "SYMPTOM" in entity_label.upper():
            structured_data["symptomes"].append(entity_text)
        elif "TREATMENT" in entity_label.upper():
            structured_data["traitements"].append(entity_text)
        elif "DIAGNOSIS" in entity_label.upper():
            structured_data["diagnostics"].append(entity_text)

    return structured_data


# Exemple de texte médical
text = (
    "Le patient présente une toux sèche persistante accompagnée de fièvre modérée. "
    "On lui a prescrit du paracétamol pour soulager la douleur. Le diagnostic probable est une infection virale."
)

# Extraire les entités et les structurer en JSON
json_output = extract_medical_entities(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
