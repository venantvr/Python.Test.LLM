import json
import re

from transformers import AutoTokenizer, AutoModel

# Charger MiniLM pour l'extraction de caractéristiques
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Définir des mots-clés pour catégoriser les informations en symptômes, traitements, diagnostics
symptoms_keywords = ["toux", "fièvre", "douleur", "maux", "sensation", "nausée", "vomissement", "fatigue"]
treatment_keywords = ["traitement", "médicament", "repos", "chirurgie", "paracétamol", "antibiotique", "prescription"]
diagnosis_keywords = ["diagnostic", "maladie", "syndrome", "inflammation", "infection"]


def classify_text(text):
    """
    Classifie un texte selon qu'il contient des symptômes, traitements ou diagnostics,
    en se basant sur des mots-clés.
    """
    if any(keyword in text.lower() for keyword in symptoms_keywords):
        return "symptome"
    elif any(keyword in text.lower() for keyword in treatment_keywords):
        return "traitement"
    elif any(keyword in text.lower() for keyword in diagnosis_keywords):
        return "diagnostic"
    return None


def extract_information(text):
    """
    Divise le texte en phrases, analyse chaque phrase avec MiniLM pour créer un JSON structuré.
    """
    sentences = re.split(r'(?<=\.)\s+', text)  # Divise le texte en phrases
    structured_data = {"symptomes": [], "traitements": [], "diagnostics": []}

    for sentence in sentences:
        category = classify_text(sentence)

        if category:
            # Encode la phrase avec MiniLM pour obtenir un embedding (facultatif ici, pour plus de détails)
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
            outputs = model(**inputs)

            # Ajouter la phrase à la catégorie correspondante dans le JSON
            entry = {"text": sentence}
            if category == "symptome":
                structured_data["symptomes"].append(entry)
            elif category == "traitement":
                structured_data["traitements"].append(entry)
            elif category == "diagnostic":
                structured_data["diagnostics"].append(entry)

    return structured_data


# Exemple de texte médical
text = ("Le patient présente une toux sèche persistante accompagnée de fièvre modérée. On lui a prescrit du "
        "paracétamol pour soulager la douleur. Le diagnostic est une infection virale bénigne.")

# Extraire les informations en JSON structuré
json_output = extract_information(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
