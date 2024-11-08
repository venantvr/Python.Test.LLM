from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le modèle et le tokenizer (en remplaçant par un modèle biomédical si disponible)
model_name = "dmis-lab/biobert-base-cased-v1.1"  # Ou un autre modèle NER biomédical
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer une pipeline NER avec `grouped_entities=True`
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Texte médical à analyser
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Effectuer la reconnaissance d'entités
entities = nlp(text)

# Post-traitement pour structurer en JSON
structured_output = []
for entity in entities:
    entity_data = {
        "text": entity["word"],
        "label": entity["entity_group"]
    }
    structured_output.append(entity_data)

# Afficher le JSON structuré
import json
print(json.dumps(structured_output, indent=2, ensure_ascii=False))
