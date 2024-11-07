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
