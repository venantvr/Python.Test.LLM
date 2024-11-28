from llama_cpp import Llama
from outlines import models, generate
from pydantic import BaseModel

llm = Llama("/models/mistral-7b-instruct-v0.2.Q6_K.gguf", n_gpu_layers=10, n_ctx=0, verbose=False)
model = models.LlamaCpp(llm)


class User(BaseModel):
    first_name: str
    last_name: str
    age: int


generator = generate.json(model, User, whitespace_pattern="")

# generator = generate.json(model, add, whitespace_pattern="")

result = generator(
    """Based on user information create a user profile with the fields first_name, last_name, age.
    User information is: Jane Doe age=10"""
)

print(result)
