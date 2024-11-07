from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle mT5-small
model_name = "google/mt5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def anonymize_with_t5(text):
    # Essayons une instruction alternative pour aider le modèle
    input_text = f"Remove personal information: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Générer plusieurs séquences pour essayer des reformulations variées
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_return_sequences=5,  # Générer davantage de séquences
        do_sample=True,
        temperature=0.9  # Ajuster la température pour une plus grande diversité
    )

    # Décoder les résultats et retourner la première reformulation valable
    for output in outputs:
        anonymized_text = tokenizer.decode(output, skip_special_tokens=True)
        if "<extra_id" not in anonymized_text:  # S'assurer que le texte est bien reformulé
            return anonymized_text

    # Si aucune reformulation n'est bonne, retourner un message par défaut
    return "Anonymisation non réussie"


# Exemple de texte
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_with_t5(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
