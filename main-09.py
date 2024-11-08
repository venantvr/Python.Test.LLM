from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Charger le modèle BioBERT pour la NER biomédicale
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer une pipeline NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Texte médical à analyser (en français)
text = """
Mme A.P., âgée de 52 ans, non fumeuse, ayant un diabète de type 2, a été hospitalisée pour une pneumopathie infectieuse.
"""

# Effectuer la reconnaissance d'entités
entities = nlp(text)
for entity in entities:
    print(f"Texte : {entity['word']}, Étiquette : {entity['entity']}")
