from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle mT5-small
model_name = "google/mt5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def anonymize_with_rewriting(text):
    # Fournir une instruction plus explicite pour une réécriture anonymisée
    input_text = f"Rewrite the sentence to anonymize sensitive information: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Générer plusieurs variantes pour augmenter les chances d'une reformulation satisfaisante
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_return_sequences=3,  # Essayer plusieurs reformulations
        do_sample=True,
        temperature=0.7  # Ajuster la température pour varier les reformulations
    )

    # Décoder et retourner la première reformulation valide
    for output in outputs:
        anonymized_text = tokenizer.decode(output, skip_special_tokens=True)
        if "<extra_id" not in anonymized_text:  # Vérifier que la reformulation est correcte
            return anonymized_text

    # Si aucune reformulation n'est bonne, retourner un message par défaut
    return "Anonymisation non réussie"


# Exemple de texte à anonymiser
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_with_rewriting(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
