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
        f"'medical_history', 'current_symptoms', 'pathology' et 'request_details'. "
        f"Répondez uniquement en JSON.\nTexte : {text}"
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
text = """
'Consultation endocrino :  il y'' à 15 ans endocrinologues, suivi par son médecin traitant. 
Consultation gynéco : il y''a 3 ans, avec échographie thyroïdienne = thyroïdite

Normalement indication d''être suivi tous les 6 mois par un endocrinologue.

Poids stable mais menstruation irrégulière, aménorrhée pendant 5 mois = la gynécologue dit que c''est surement lié aux hormones ou stress

Pas de lettre de recommandation 
Pas de dernier bilan sanguin thyroïdien'
"""

# Générer le JSON structuré
json_output = generate_structured_json(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
