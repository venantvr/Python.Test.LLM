import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Remplacez "YOUR_API_KEY" par votre token Hugging Face
api_key = "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"
model_name = "mistralai/Ministral-8B-Instruct-2410"

# Charger le modèle et le tokenizer avec le token d'API
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key)


def generate_structured_json(text: str) -> dict:
    # Préparer le prompt pour le modèle
    prompt = (
        f"Veuillez analyser le texte médical suivant et fournir une sortie JSON structurée avec les champs "
        f"'symptomes', 'traitements', 'diagnostics' et 'suivi'. Répondez uniquement en JSON.\n\nTexte : {text}"
    )

    # Tokenizer le prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

    # Générer la réponse
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    # Décoder la sortie
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Convertir en dictionnaire Python
    try:
        structured_data = json.loads(json_response)
    except json.JSONDecodeError:
        print("Erreur lors de la conversion en JSON. Format non conforme.")
        structured_data = {"error": "Format JSON incorrect"}

    return structured_data


# Exemple de texte médical
text = (
    "Le patient de 65 ans se présente avec une toux productive, de la fièvre élevée (39,5 °C), des frissons et une "
    "douleur thoracique augmentée à l'inspiration. Il a également signalé un essoufflement et une fatigue importante. "
    "Lors de l'auscultation, des râles crépitants et une diminution des bruits respiratoires sont notés dans le lobe "
    "inférieur droit. Une radiographie pulmonaire révèle une opacité dans le même lobe, confirmant la suspicion de "
    "pneumonie bactérienne. Un traitement par antibiotiques est immédiatement prescrit, en commençant par une "
    "administration de ceftriaxone intraveineuse, complétée par de l'azithromycine orale pour couvrir les germes "
    "atypiques. Le patient est également hydraté par voie intraveineuse et reçoit de l'oxygène pour maintenir une "
    "saturation en oxygène supérieure à 92 %. Des analyses de sang montrent une élévation des marqueurs inflammatoires, "
    "y compris la CRP et les globules blancs, ce qui renforce le diagnostic. Une culture des expectorations est en cours "
    "pour identifier le pathogène exact et ajuster l'antibiothérapie si nécessaire. Le patient est suivi de près pour "
    "évaluer la réponse au traitement, avec une amélioration attendue dans les 48 à 72 heures. En cas de complications, "
    "telles qu'un abcès pulmonaire ou une détérioration de l'état respiratoire, un transfert en soins intensifs serait "
    "envisagé."
)

# Générer le JSON structuré
json_output = generate_structured_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
