import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Charger le modèle et le tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_json_from_text(text: str) -> dict:
    prompt = (
        f"Analyse le texte suivant et retourne un JSON structuré avec les champs "
        f"'symptômes', 'traitements', 'diagnostics'. Réponds uniquement en JSON : {text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        structured_data = json.loads(json_response)
    except json.JSONDecodeError:
        structured_data = {"error": "Format JSON incorrect"}
    return structured_data


# Exemple de texte
text = "Le patient présente une toux sèche persistante accompagnée de fièvre modérée. On lui a prescrit du paracétamol pour soulager la douleur."
json_output = generate_json_from_text(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
