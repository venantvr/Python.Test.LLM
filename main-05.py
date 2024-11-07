import json
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Charger le modèle et le tokenizer FLAN-T5 en local
model_name = "google/flan-t5-large"  # Utilisez 'flan-t5-small' si vous avez moins de ressources
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def extract_json_part(text):
    """
    Fonction pour extraire uniquement la partie JSON de la réponse.
    Utilise des expressions régulières pour détecter le bloc JSON.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def generate_medical_json(text: str) -> dict:
    # Préparer le prompt pour le modèle avec un exemple de JSON attendu
    prompt = (
        f"Analyse le texte suivant et retourne uniquement un JSON strict structuré avec les champs 'symptomes', "
        f"'traitements', 'diagnostics'. Voici un exemple de format attendu : "
        f"{{'symptomes': ['toux sèche', 'fièvre modérée'], 'traitements': ['paracétamol'], 'diagnostics': []}}. "
        f"Texte : {text}"
    )

    # Tokenizer et génération de la réponse
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)

    # Décoder le tenseur en texte
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Réponse brute du modèle:", json_response)

    # Extraire uniquement la partie JSON
    json_response = extract_json_part(json_response)

    if json_response:
        try:
            # Convertir la réponse en JSON
            structured_data = json.loads(json_response)
        except json.JSONDecodeError:
            print("Erreur lors de la conversion en JSON. Vérifiez la sortie du modèle.")
            structured_data = {"error": "Format JSON incorrect"}
    else:
        structured_data = {"error": "Aucun JSON détecté"}

    return structured_data


# Exemple de texte médical
text = ("Le patient présente une toux sèche persistante accompagnée de fièvre modérée. On lui a prescrit du "
        "paracétamol pour soulager la douleur.")

# Générer le JSON structuré
json_output = generate_medical_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
