from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Load the mT5-small model and MT5Tokenizer
model_name = "google/mt5-small"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)


def anonymize_with_rewriting(text):
    # Provide a more explicit instruction for anonymized rewriting
    input_text = f"Rewrite the sentence to anonymize sensitive information: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate multiple variants to increase the chances of a satisfactory reformulation
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_return_sequences=3,  # Generate multiple reformulations
        do_sample=True,
        temperature=0.7  # Adjust temperature for diversity in reformulations
    )

    # Decode and return the first valid reformulation
    for output in outputs:
        anonymized_text = tokenizer.decode(output, skip_special_tokens=True)
        if "<extra_id" not in anonymized_text:  # Ensure a correct reformulation
            return anonymized_text

    # Default message if no valid reformulation is produced
    return "Anonymisation non réussie"


# Example text to anonymize
text = "Jean Dupont habite à Paris depuis 2021. Son numéro de téléphone est 0123456789."
anonymized_text = anonymize_with_rewriting(text)

print("Texte original :", text)
print("Texte anonymisé :", anonymized_text)
