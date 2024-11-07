import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Charger flan-t5 en local
model_name = "google/flan-t5-small"  # Utilisez une version plus grande (base/large) si votre machine le permet
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_simplified_medical_json(text: str) -> dict:
    # Prompt pour obtenir un résumé structuré
    prompt = (
        f"Simplifie et retourne uniquement un JSON structuré pour ce texte médical. Utilise les catégories suivantes : "
        f"'symptomes', 'traitements', 'diagnostics', et 'suivi'. Répond uniquement en JSON.\n\nTexte : {text}"
    )

    # Encoder le prompt et générer la réponse
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)

    # Décoder la sortie en texte JSON
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        # Convertir en JSON
        structured_data = json.loads(json_response)
    except json.JSONDecodeError:
        print("Erreur lors de la conversion en JSON. Format non conforme.")
        structured_data = {"error": "Format JSON incorrect"}

    return structured_data


# Texte médical plus complexe pour tester le modèle
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
