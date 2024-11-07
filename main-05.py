import json

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le modèle NER français (CamemBERT NER)
model_name = "Jean-Baptiste/camembert-ner-with-lower-case"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer le pipeline de NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def extract_medical_entities(text: str) -> dict:
    # Extraire les entités nommées
    entities = ner_pipeline(text, truncation=True, max_length=512)

    # Initialiser des listes pour chaque catégorie
    structured_data = {"symptomes": [], "traitements": [], "diagnostics": []}

    # Classifier les entités extraites
    for entity in entities:
        entity_text = entity['word']
        entity_label = entity['entity']

        # Classifications basées sur des mots-clés dans les labels (simplifié)
        if "MALADIE" in entity_label.upper() or "SYMPTOM" in entity_label.upper():
            structured_data["symptomes"].append(entity_text)
        elif "MEDICAMENT" in entity_label.upper() or "DRUG" in entity_label.upper():
            structured_data["traitements"].append(entity_text)
        elif "DIAGNOSTIC" in entity_label.upper():
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
