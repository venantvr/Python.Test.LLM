import re

from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle distilGPT-2
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def anonymize_text(text):
    # Utiliser des expressions régulières pour anonymiser les noms et dates
    text = re.sub(r'\b[A-Z][a-z]+\b', 'Personne', text)  # Remplacer les noms propres par 'Personne'
    text = re.sub(r'\b\d{2,4}-\d{2,4}\b', 'Date', text)  # Remplacer les dates par 'Date'
    text = re.sub(r'\b\d{5}\b', 'CodePostal', text)  # Remplacer les codes postaux par 'CodePostal'

    # Utiliser le modèle pour reformuler légèrement le texte anonymisé
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=256, num_return_sequences=1, do_sample=True)
    anonymized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return anonymized_text


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_text(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
