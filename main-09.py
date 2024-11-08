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
                "age": {"type": "integer"},
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
