from gliner import GLiNER


def camembert_bio_gliner(text):
    model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")

    # labels = ["Âge", "Patient", "Maladie", "Symptômes"]
    labels = ["Âge", "Patient", "Maladie", "Symptômes", "Habitude de vie", "Antécédent médical"]

    entities = model.predict_entities(text, labels, threshold=0.5, flat_ner=True)

    for entity in entities:
        print(entity["text"], "=>", entity["label"])


if __name__ == "__main__":
    text = """
    Mme A.P. âgée de 52 ans, non tabagique, ayant un diabète de type 2 a été hospitalisée pour une 
    pneumopathie infectieuse. Cette patiente présentait depuis 2 ans des infections respiratoires traités en ambulatoire. 
    L’examen physique a trouvé une fièvre à 38ºc et un foyer de râles crépitants de la base pulmonaire droite.
    """

    camembert_bio_gliner(text)
