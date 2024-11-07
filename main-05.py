import json
import re
from transformers import AutoTokenizer, AutoModel
import torch

# Charger MiniLM pour l'extraction de caractéristiques
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Mots-clés pour classer les informations
symptoms_keywords = ["toux", "fièvre", "frissons", "douleur thoracique", "essoufflement", "fatigue"]
treatment_keywords = ["antibiotiques", "ceftriaxone", "azithromycine", "hydratation", "oxygène"]
diagnosis_keywords = ["pneumonie", "opacité", "inflammation", "CRP", "globules blancs"]
followup_keywords = ["suivi", "intensifs", "complications"]


def classify_text(text):
    """
    Classifie un texte en symptômes, traitements, diagnostics, ou suivi selon les mots-clés.
    """
    if any(keyword in text.lower() for keyword in symptoms_keywords):
        return "symptome"
    elif any(keyword in text.lower() for keyword in treatment_keywords):
        return "traitement"
    elif any(keyword in text.lower() for keyword in diagnosis_keywords):
        return "diagnostic"
    elif any(keyword in text.lower() for keyword in followup_keywords):
        return "suivi"
    return None


def extract_information(text):
    """
    Divise le texte en phrases, analyse chaque phrase avec MiniLM pour créer un JSON structuré.
    """
    sentences = re.split(r'(?<=\.)\s+', text)  # Divise le texte en phrases
    structured_data = {
        "symptomes": [],
        "traitements": [],
        "diagnostics": [],
        "suivi": []
    }

    for sentence in sentences:
        category = classify_text(sentence)

        if category:
            # Ajouter la phrase à la catégorie correspondante dans le JSON
            entry = {"text": sentence}
            if category == "symptome":
                structured_data["symptomes"].append(entry)
            elif category == "traitement":
                structured_data["traitements"].append(entry)
            elif category == "diagnostic":
                structured_data["diagnostics"].append(entry)
            elif category == "suivi":
                structured_data["suivi"].append(entry)

    return structured_data


# Texte médical complexe à analyser
text = (
    "Le patient de 65 ans se présente avec une toux productive, de la fièvre élevée (39,5 °C), des frissons et une "
    "douleur thoracique augmentée à l'inspiration. Il a également signalé un essoufflement et une fatigue importante. "
    "Lors de l'auscultation, des râles crépitants et une diminution des bruits respiratoires sont notés dans le lobe "
    "inférieur droit. Une radiographie pulmonaire révèle une opacité dans le même lobe, confirmant la suspicion de "
    "pneumonie bactérienne. Un traitement par antibiotiques est immédiatement prescrit, en commençant par une "
    "administration de ceftriaxone intraveineuse, complétée par de l'azithromycine orale pour couvrir les germes "
    "atypiques. Le patient est également hydraté par voie intraveineuse et reçoit de l'oxygène pour maintenir une "
    "saturation en oxygène supérieure à 92 %. Des analyses de sang montrent une élévation des marqueurs inflammatoires, "
    "y compris la CRP et les globules blancs, ce qui renforce le diagnostic. Une culture des expectorations est en cours "
    "pour identifier le pathogène exact et ajuster l'antibiothérapie si nécessaire. Le patient est suivi de près pour "
    "évaluer la réponse au traitement, avec une amélioration attendue dans les 48 à 72 heures. En cas de complications, "
    "telles qu'un abcès pulmonaire ou une détérioration de l'état respiratoire, un transfert en soins intensifs serait "
    "envisagé."
)

# Extraire les informations en JSON structuré
json_output = extract_information(text)
print(json.dumps(json_output, indent=4, ensure_ascii=False))
