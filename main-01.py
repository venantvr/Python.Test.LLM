from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle mT5-small
model_name = "google/mt5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def anonymize_with_t5(text):
    # Préparer le texte en entrée en indiquant une tâche de paraphrase
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Générer le texte paraphrasé
    outputs = model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1, do_sample=True)
    anonymized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return anonymized_text


# Exemple de texte
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_with_t5(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
