import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
