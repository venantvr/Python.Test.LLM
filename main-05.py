import json

from transformers import AutoTokenizer, AutoModelForCausalLM

# Utiliser distilgpt2 pour réduire la consommation de mémoire
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_simplified_medical_json(text: str) -> dict:
    # Prompt avec exemple pour structurer la sortie en JSON
    prompt = (
        f"Simplifie et structure le texte médical suivant en JSON avec les catégories 'symptomes', 'traitements', "
        f"'diagnostics', et 'suivi'. Répond uniquement en JSON.\n\n"
        f"Texte : {text}\n\n"
    )

    # Tokenizer avec max_length pour éviter les dépassements de mémoire
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=200, num_beams=2, early_stopping=True)

    # Décoder la sortie en texte JSON
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Réponse brute du modèle:", json_response)

    # Essayer de convertir en JSON
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

# Générer le JSON structuré et simplifié
json_output = generate_simplified_medical_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
