from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Charger le modèle NER pour détecter les entités et comprendre les rôles
ner_model = "Jean-Baptiste/camembert-ner"
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_model, aggregation_strategy="simple")

# Charger FLAN-T5 pour la reformulation et la contextualisation
model_name = "google/flan-t5-small"  # Modèle T5 spécialisé pour le suivi d'instructions
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def detect_entities_and_roles(text):
    # Détection des entités et attribution de rôles spécifiques
    entities = ner_pipeline(text)
    anonymized_text = text
    roles = []

    for entity in entities:
        role = ""
        if entity['entity_group'] == "PER":
            role = "résident"
            anonymized_text = anonymized_text.replace(entity['word'], "Personne")
        elif entity['entity_group'] == "LOC":
            role = "lieu de résidence"
            anonymized_text = anonymized_text.replace(entity['word'], "Lieu")
        elif entity['entity_group'] == "ORG":
            role = "entreprise"
            anonymized_text = anonymized_text.replace(entity['word'], "Organisation")
        elif entity['entity_group'] == "MISC" or entity['entity_group'] == "DATE":
            role = "date ou événement"
            anonymized_text = anonymized_text.replace(entity['word'], "Date")
        roles.append(f"{entity['word']} est un {role}")

    return anonymized_text, roles


def rewrite_with_roles(text, roles):
    # Instruction explicite pour reformuler en tenant compte des rôles des entités
    input_text = f"Reformule le texte suivant en tenant compte des rôles : {', '.join(roles)} : {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Générer le texte reformulé
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, do_sample=True)

    # Décoder le texte généré
    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten_text


def chatbot():
    print("Bienvenue dans l'agent d'anonymisation avec compréhension des rôles. Tapez 'exit' pour quitter.")
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == "exit":
            print("Agent : Au revoir!")
            break

        # Détecter les entités et leurs rôles
        anonymized_text, roles = detect_entities_and_roles(user_input)

        # Reformuler en tenant compte des rôles contextuels
        rewritten_text = rewrite_with_roles(anonymized_text, roles)

        print("Agent (texte anonymisé et reformulé) :", rewritten_text)


# Lancer le chatbot
chatbot()
