from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from llms.biobert_mnli_snli_scinli_scitail_mednli_stsb import biobert_mnli_snli_scinli_scitail_mednli_stsb
from llms.camembert_base import camembert_base
from llms.camembert_bio_generalized import camembert_bio_generalized
from llms.paraphrase_multilingual_minilm import paraphrase_multilingual_minilm

app = FastAPI()


# Définir le modèle de données attendu pour l'entrée
class SimilarityRequest(BaseModel):
    method: str
    phrase1: str
    phrase2: str


@app.post("/similarity")
async def get_similarity_score(data: SimilarityRequest) -> Dict[str, float]:
    match data.method:
        case "paraphrase_multilingual_minilm":
            score = paraphrase_multilingual_minilm(data.phrase1, data.phrase2)
        case "camembert_base":
            score = camembert_base(data.phrase1, data.phrase2)
        case "camembert_bio_generalized":
            score = camembert_bio_generalized(data.phrase1, data.phrase2)
        case "biobert_mnli_snli_scinli_scitail_mednli_stsb":
            score = biobert_mnli_snli_scinli_scitail_mednli_stsb(data.phrase1, data.phrase2)
        case _:
            score = 0.0

    # Appel de la méthode `process` pour obtenir le score de similarité
    # score = process(data.phrase1, data.phrase2)
    return {"similarity_score": score}


# Point de lancement de l'application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
