import json

from transformers import AutoModelForCausalLM, AutoTokenizer

access_token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"

# Charger le tokenizer et le modèle avec le token d'authentification
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)


def generate_medical_json(text: str) -> dict:
    prompt = (
        f"Analyse le texte médical suivant et retourne un JSON structuré avec les champs "
        f"'symptomes', 'traitements', 'diagnostics': {text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)

    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
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
