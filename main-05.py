from gliner import GLiNER

model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")

# text = """
# Mme A.P. âgée de 52 ans, non tabagique, ayant un diabète de type 2 a été hospitalisée pour une
# pneumopathie infectieuse. Cette patiente présentait depuis 2 ans des infections respiratoires traités en ambulatoire.
# L’examen physique a trouvé une fièvre à 38ºc et un foyer de râles crépitants de la base pulmonaire droite.
# """

text = """Bonjour,
Je cherche un spécialiste en dermatologie officiant dans une structure bien équipée afin de pratiquer des actes de chirurgie tel que le retrait de grains de beauté (non cancéreux) sans laisser de cicatrices...
Cordialement,
SH"""

# labels = ["Âge", "Patient", "Maladie", "Symptômes"]
labels = ["Âge", "Patient", "Maladie", "Symptômes", "Habitude de vie", "Antécédent médical"]

entities = model.predict_entities(text, labels, threshold=0.1, flat_ner=True)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
