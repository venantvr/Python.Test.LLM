from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle T5 en local
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Texte d'entrée et prompt pour obtenir du JSON structuré
input_text = """
Patient nommé Jean Dupont, 52 ans, homme, avec des antécédents d'hypertension et de diabète de type 2. 
Récemment, il a ressenti des douleurs thoraciques sévères lors d'exercices physiques et un essoufflement 
modéré même au repos. Il a été diagnostiqué avec une maladie coronarienne le 1er juin 2024 et cherche 
une consultation ainsi qu'un plan de traitement, avec une préférence pour un rendez-vous le 1er juillet 2024.
"""
prompt = (
    "Veuillez convertir le texte suivant en JSON structuré au format suivant :\n"
    "{\n"
    "  \"request_summary\": {\n"
    "    \"patient_info\": {\n"
    "      \"name\": \"\",\n"
    "      \"age\": 0,\n"
    "      \"gender\": \"\"\n"
    "    },\n"
    "    \"medical_history\": [\n"
    "      {\"condition\": \"\", \"diagnosed_date\": \"\", \"treatment\": \"\"}\n"
    "    ],\n"
    "    \"current_symptoms\": [\n"
    "      {\"symptom\": \"\", \"description\": \"\", \"duration\": \"\", \"severity\": \"\"}\n"
    "    ],\n"
    "    \"pathology\": {\n"
    "      \"name\": \"\",\n"
    "      \"diagnosed_date\": \"\",\n"
    "      \"previous_treatments\": \"\"\n"
    "    },\n"
    "    \"request_details\": {\n"
    "      \"purpose\": \"\",\n"
    "      \"preferred_appointment_date\": \"\",\n"
    "      \"additional_notes\": \"\"\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    f"Texte : {input_text}"
)

# Encoder le texte et générer la sortie
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=512)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
