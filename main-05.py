from gliner import GLiNER

model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")

# text = """
# Mme A.P. âgée de 52 ans, non tabagique, ayant un diabète de type 2 a été hospitalisée pour une
# pneumopathie infectieuse. Cette patiente présentait depuis 2 ans des infections respiratoires traités en ambulatoire.
# L’examen physique a trouvé une fièvre à 38ºc et un foyer de râles crépitants de la base pulmonaire droite.
# """

# text = """Bonjour,
# Je cherche un spécialiste en dermatologie officiant dans une structure bien équipée afin de pratiquer des actes de chirurgie tel que le retrait de grains de beauté (non cancéreux) sans laisser de cicatrices...
# Cordialement,
# SH"""

# text = """
# Bonjour,
# Ma fille a un très gros rhume des foins, Nez qui coule et les yeux qui pleurent,  je lui donne du Zirtex et du spray nasal. mais cela ne la soulage pas longtemps.
# Je recherche un allergologue pour lui faire des tests et trouver une solution pour la soulagée. Je vous remercie d'avance, Mme Michu
# """

text = """
'Hello l''équipe,

Notre fille de 8 mois pleure plus d''une dizaine de fois / nuit.
Nous aimerions consulter "LE" spécialiste du sommeil pour les bébés. 
Pédiatre spécialisé dans le sommeil ? Autres spécialistes ? Il me semble qu''il y''a un centre (HP) qui a un service dédié à ce sujet.

Un grand merci par avance 
"""

# labels = ["Âge", "Patient", "Maladie", "Symptômes"]
labels = ["Âge", "Patient", "Maladie", "Symptômes", "Traitement", "Habitude de vie", "Antécédent médical", "Spécialité"]

entities = model.predict_entities(text, labels, threshold=0.1, flat_ner=True)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
