# inference_server.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI(title="Smart Home Semantic AI")

# Load database with precomputed embeddings
with open("database_precomputed.json", "r", encoding="utf-8") as f:
    db_data = json.load(f)

# Load model (only for query encoding)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class QueryRequest(BaseModel):
    message: str

@app.post("/query")
def query(request: QueryRequest):
    try:
        query_emb = model.encode(request.message, convert_to_tensor=True)
        similarities = [util.cos_sim(query_emb, torch.tensor(e["embedding"])) for e in db_data]
        best_idx = similarities.index(max(similarities))
        return JSONResponse({"ai_reply": db_data[best_idx]["content"]})
    except Exception as e:
        return JSONResponse({"ai_reply": f"[Error] {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
