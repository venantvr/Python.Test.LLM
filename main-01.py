from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle mT5-small
model_name = "google/mt5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def anonymize_with_t5(text):
    # Fournir une instruction plus précise pour l'anonymisation
    input_text = f"anonymize: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Générer plusieurs séquences pour augmenter les chances d'une reformulation efficace
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_return_sequences=3,  # Essayer plusieurs reformulations
        do_sample=True,
        temperature=0.7  # Ajuster la température pour varier les reformulations
    )

    # Décoder les résultats et retourner le premier texte reformulé valide
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
