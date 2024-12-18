from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle et le tokenizer de Hugging Face
model_name = "facebook/opt-350m"  # Modèle léger
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajouter un token spécial et redimensionner les embeddings du modèle
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Définir le schéma JSON pour structurer la sortie avec une liste
schema = {
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "price": {"type": "number"}
                }
            }
        }
    }
}

# Texte d'entrée
input_text = "Voici une liste de produits : un smartphone à 699.99 euros, un ordinateur portable à 999.99 euros, et des écouteurs à 59.99 euros."

# Initialiser Jsonformer
jsonformer = Jsonformer(
    model=model,
    tokenizer=tokenizer,
    json_schema=schema,
    prompt=input_text  # Ajoute le texte d'entrée comme prompt
)

# Générer et structurer le JSON avec une liste
structured_json = jsonformer()
print(structured_json)
