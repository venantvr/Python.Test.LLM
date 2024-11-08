from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Charger le tokenizer et le modèle AliBERT
tokenizer = AutoTokenizer.from_pretrained("AliBERT")
model = AutoModelForTokenClassification.from_pretrained("AliBERT")

# Créer une pipeline NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Texte médical à analyser
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Effectuer la reconnaissance d'entités
entities = nlp(text)
for entity in entities:
    print(f"Texte : {entity['word']}, Étiquette : {entity['entity']}")
