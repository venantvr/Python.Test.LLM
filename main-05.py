import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Charger le modèle et le tokenizer FLAN-T5 en local
model_name = "google/flan-t5-large"  # Utilisez 'flan-t5-small' si vous avez moins de ressources
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_medical_json(text: str) -> dict:
    # Préparer le prompt pour le modèle
    prompt = (
        f"Analyse le texte médical suivant et retourne un JSON structuré avec les champs "
        f"'symptomes', 'traitements', 'diagnostics'. Réponds uniquement en JSON : {text}"
    )

    # Tokenizer et génération de la réponse
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)

    print(outputs[0])

    # Décoder et structurer la réponse
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        # Convertir la réponse en JSON
        structured_data = json.loads(json_response)
    except json.JSONDecodeError:
        print("Erreur lors de la conversion en JSON. Vérifiez la sortie du modèle.")
        structured_data = {"error": "Format JSON incorrect"}

    return structured_data


# Exemple de texte médical
text = ("Le patient présente une toux sèche persistante accompagnée de fièvre modérée. On lui a prescrit du "
        "paracétamol pour soulager la douleur.")

# Générer le JSON structuré
json_output = generate_medical_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
