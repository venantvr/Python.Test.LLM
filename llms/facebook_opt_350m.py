from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle et le tokenizer de Hugging Face
model_name = "facebook/opt-350m"  # Modèle léger
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Définir le schéma JSON pour structurer la sortie avec une liste
schema = {
    "products": [
        {
            "product_name": "string",
            "price": "float"
        }
    ]
}

# Texte d'entrée
input_text = "Voici une liste de produits : un smartphone à 699.99 euros, un ordinateur portable à 999.99 euros, et des écouteurs à 59.99 euros."

# Initialiser Jsonformer
jsonformer = Jsonformer(
    model=model,
    tokenizer=tokenizer,
    schema=schema,
    prompt=input_text  # Ajoute le texte d'entrée comme prompt
)

# Générer et structurer le JSON avec une liste
structured_json = jsonformer()
print(structured_json)
