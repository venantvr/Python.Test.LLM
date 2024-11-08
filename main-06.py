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
    json = """
    {
  "request_summary": {
    "patient_info": {
      "name": "John Doe",
      "age": 45,
      "gender": "Male"
    },
    "medical_history": [
      {
        "condition": "Hypertension",
        "diagnosed_date": "2015-06-15",
        "treatment": "Lisinopril 10 mg daily"
      }
    ],
    "current_symptoms": [
      {
        "symptom": "Chest pain",
        "description": "Sharp pain in the chest, occurring primarily during physical exertion.",
        "duration": "2 weeks",
        "severity": "Severe"
      }
    ],
    "pathology": {
      "name": "Coronary Artery Disease",
      "diagnosed_date": "2024-06-01",
      "previous_treatments": "None"
    },
    "request_details": {
      "purpose": "Consultation and Treatment Plan",
      "preferred_appointment_date": "2024-07-01",
      "additional_notes": "Seeking a comprehensive evaluation ..."
    }
  }
}
    """

    # Préparer le prompt pour le modèle
    prompt = (
        f"Veuillez analyser le texte médical suivant et fournir une sortie JSON structurée au format {json} "
        f"Répondez uniquement en JSON.\n\nTexte : {text}"
    )

    # Tokenizer le prompt avec attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs["attention_mask"] = torch.ones(inputs.input_ids.shape, dtype=torch.long)

    # Spécifier pad_token_id pour éviter l’avertissement
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id  # Utiliser le token de fin comme pad_token_id
        )

    # Décoder la sortie
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Convertir en dictionnaire Python
    try:
        structured_data = json.loads(json_response)
    except json.JSONDecodeError:
        print("Erreur lors de la conversion en JSON. Format non conforme.")
        structured_data = {"error": "Format JSON incorrect"}
        print(json_response)

    return structured_data


# Exemple de texte médical
text = (
    "Le patient de 65 ans se présente avec une toux productive, de la fièvre élevée (39,5 °C), des frissons et une "
    "douleur thoracique augmentée à l'inspiration. Il a également signalé un essoufflement et une fatigue importante. "
    "Lors de l'auscultation, des râles crépitants et une diminution des bruits respiratoires sont notés dans le lobe "
    "inférieur droit. Une radiographie pulmonaire révèle une opacité dans le même lobe, confirmant la suspicion de "
    "pneumonie bactérienne. Un traitement par antibiotiques est immédiatement prescrit, en commençant par une "
    "administration de ceftriaxone intraveineuse, complétée par de l'azithromycine orale pour couvrir les germes "
    "atypiques."
)

# Générer le JSON structuré
json_output = generate_structured_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
