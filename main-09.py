import json

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger BioBERT en local pour une tâche NER biomédicale
model_name = "dmis-lab/biobert-base-cased-v1.1"  # Modèle biomédical pré-entraîné
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer une pipeline NER avec `grouped_entities=True`
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Texte médical en français à analyser
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Utiliser la pipeline NER
entities = nlp(text)

# Définir un schéma JSON personnalisé pour les entités médicales
structured_output = {
    "Patient": [],
    "Âge": [],
    "Antécédents médicaux": [],
    "Maladies actuelles": [],
    "Symptômes": []
}

# Définir un mapping des labels vers les catégories du schéma
label_map = {
    "PER": "Patient",
    "AGE": "Âge",
    "DISEASE": "Maladies actuelles",
    "SYMPTOM": "Symptômes",
    "MEDICAL_HISTORY": "Antécédents médicaux"
}

# Traiter chaque entité en fonction de son label
for entity in entities:
    label = entity["entity_group"]
    text_value = entity["word"]

    # Utiliser le mapping pour structurer en JSON selon les entités
    json_key = label_map.get(label, None)
    if json_key:
        structured_output[json_key].append(text_value)

# Afficher la sortie JSON structurée
print(json.dumps(structured_output, indent=2, ensure_ascii=False))
