from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle et le tokenizer T5
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Texte d'entrée décrivant une problématique médicale
input_text = "Patient nommé Jean Dupont, 52 ans, homme, avec des antécédents d'hypertension et de diabète de type 2. Récemment, il a ressenti des douleurs thoraciques sévères lors d'exercices physiques et un essoufflement modéré même au repos. Il a été diagnostiqué avec une maladie coronarienne le 1er juin 2024 et cherche une consultation ainsi qu'un plan de traitement, avec une préférence pour un rendez-vous le 1er juillet 2024."

# Préparer l'entrée pour le modèle
input_ids = tokenizer.encode(f"traduire en JSON : {input_text}", return_tensors="pt")

# Générer la sortie
output_ids = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
