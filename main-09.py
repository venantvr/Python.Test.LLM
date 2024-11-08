import json

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger un modèle multilingue pour NER qui fonctionne bien avec le français
model_name = "Davlan/xlm-roberta-large-ner-hrl"  # Modèle NER multilingue qui fonctionne bien pour le français
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer une pipeline NER avec `grouped_entities=True`
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Texte médical à analyser
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Utiliser la pipeline NER
entities = nlp(text)

# Post-traitement pour transformer la sortie en JSON structuré
structured_output = {"Patient": [], "Âge": [], "Maladie": [], "Symptôme": []}

# Mapper les labels NER vers les clés de votre JSON
label_map = {
    "PER": "Patient",  # Personne identifiée
    "AGE": "Âge",  # Entité d'âge si détectable
    "DISEASE": "Maladie",  # Pour les maladies détectées
    "SYMPTOM": "Symptôme",  # Pour les symptômes médicaux
}

# Filtrer et organiser les entités dans le JSON structuré
for entity in entities:
    label = entity["entity_group"]
    text_value = entity["word"]

    # Utiliser le label map pour structurer en JSON selon les entités pertinentes
    json_key = label_map.get(label, None)
    if json_key:
        structured_output[json_key].append(text_value)

# Afficher la sortie JSON structurée
print(json.dumps(structured_output, indent=2, ensure_ascii=False))
