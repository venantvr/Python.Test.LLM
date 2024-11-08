from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle BLOOM et le tokenizer
model_name = "bigscience/bloom-1b7"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)

# Définir un token de remplissage (pad_token) pour éviter le conflit avec eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Définir le schéma JSON pour structurer la sortie
json_schema = {
    "type": "object",
    "properties": {
        "request_summary": {
            "type": "object",
            "properties": {
                "patient_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                        "gender": {"type": "string"}
                    }
                },
                "medical_history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {"type": "string"},
                            "diagnosed_date": {"type": "string", "format": "date"},
                            "treatment": {"type": "string"}
                        }
                    }
                },
                "current_symptoms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symptom": {"type": "string"},
                            "description": {"type": "string"},
                            "duration": {"type": "string"},
                            "severity": {"type": "string"}
                        }
                    }
                },
                "pathology": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "diagnosed_date": {"type": "string", "format": "date"},
                        "previous_treatments": {"type": "string"}
                    }
                },
                "request_details": {
                    "type": "object",
                    "properties": {
                        "purpose": {"type": "string"},
                        "preferred_appointment_date": {"type": "string", "format": "date"},
                        "additional_notes": {"type": "string"}
                    }
                }
            }
        }
    }
}

# Texte médical en français décrivant la problématique médicale
medical_text = """
Patient nommé Jean Dupont, 52 ans, homme, avec des antécédents d'hypertension et de diabète de type 2. Récemment, il a ressenti des douleurs thoraciques sévères lors d'exercices physiques et un essoufflement modéré même au repos. Il a été diagnostiqué avec une maladie coronarienne le 1er juin 2024 et cherche une consultation ainsi qu'un plan de traitement, avec une préférence pour un rendez-vous le 1er juillet 2024.
"""

# Encoder le texte d'entrée avec padding, attention_mask et troncature pour contrôler la longueur
inputs = tokenizer(
    medical_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,  # Ajustez cette longueur selon les besoins
    attention_mask=True
)

# Initialiser Jsonformer avec le modèle, le tokenizer, le schéma JSON et le texte médical
jsonformer = Jsonformer(model, tokenizer, json_schema, medical_text)

# Générer les données structurées en JSON
try:
    generated_data = jsonformer()
    print(generated_data)
except RuntimeError as e:
    print(f"Erreur lors de la génération : {e}")