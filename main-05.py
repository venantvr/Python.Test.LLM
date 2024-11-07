from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le modèle et le tokenizer pour l'analyse biomédicale en français
model_name = "almanach/camembert-bio-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer un pipeline pour la tâche de NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Exemple de texte médical
text = (
    "Le patient présente une toux sèche persistante accompagnée de fièvre modérée. "
    "On lui a prescrit du paracétamol pour soulager la douleur. Le diagnostic probable est une infection virale."
)

# Utiliser le pipeline pour extraire les entités
entities = ner_pipeline(text)

# Afficher les entités extraites
print("Entités extraites:", entities)

import json


def categorize_entities(entities):
    structured_data = {"symptomes": [], "traitements": [], "diagnostics": []}

    for entity in entities:
        entity_text = entity['word']
        entity_label = entity['entity']

        # Ajouter une logique simple pour classer les entités
        if "SYMPTOM" in entity_label.upper() or "ANATOMY" in entity_label.upper():
            structured_data["symptomes"].append(entity_text)
        elif "TREATMENT" in entity_label.upper() or "DRUG" in entity_label.upper():
            structured_data["traitements"].append(entity_text)
        elif "DIAGNOSIS" in entity_label.upper() or "DISEASE" in entity_label.upper():
            structured_data["diagnostics"].append(entity_text)

    return structured_data


# Structurer les entités en JSON
json_output = categorize_entities(entities)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
