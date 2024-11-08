import json
import re

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Charger un modèle NER multilingue (ou spécifique français)
model_name = "Davlan/xlm-roberta-large-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Pipeline NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Texte médical en français
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Utiliser le modèle NER
entities = nlp(text)

# Schéma JSON
structured_output = {
    "Patient": [],
    "Âge": [],
    "Antécédents médicaux": [],
    "Maladies actuelles": [],
    "Symptômes": []
}

# Extraction de l'âge avec une expression régulière
age_match = re.search(r"âgé[e]* de (\d+) ans", text)
if age_match:
    structured_output["Âge"].append(age_match.group(1))

# Extraction d'antécédents médicaux potentiels (exemple simplifié)
medical_history_matches = re.findall(r"diabète|tabagique|hypertension|asthme", text, re.IGNORECASE)
structured_output["Antécédents médicaux"].extend(medical_history_matches)

# Traitement des entités NER pour ajouter dans le JSON
label_map = {
    "PER": "Patient",
    "DISEASE": "Maladies actuelles",
    "SYMPTOM": "Symptômes"
}

for entity in entities:
    label = entity["entity_group"]
    text_value = entity["word"]
    json_key = label_map.get(label, None)
    if json_key:
        structured_output[json_key].append(text_value)

# Résultat structuré en JSON
print(json.dumps(structured_output, indent=2, ensure_ascii=False))
