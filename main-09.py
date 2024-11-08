from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tiiuae/falcon-7b"  # Exemple avec Falcon 7B
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

json_schema = {
    "type": "object",
    "properties": {
        "patient_info": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},  # Remplacer "integer" par "string"
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

input_text = """
Patient nommé Jean Dupont, 52 ans, homme, avec des antécédents d'hypertension et de diabète de type 2. 
Récemment, il a ressenti des douleurs thoraciques sévères lors d'exercices physiques et un essoufflement.
"""

jsonformer = Jsonformer(model, tokenizer, json_schema, input_text)
generated_json = jsonformer()
print(generated_json)
