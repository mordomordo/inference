# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import torch
import os

# ================= FastAPI =================
app = FastAPI(title="Smart Home QA Inference Server")

# ================= Load precomputed embeddings =================
EMBED_FILE = "precomputed_embeddings.pt"
if not os.path.exists(EMBED_FILE):
    raise FileNotFoundError(f"{EMBED_FILE} not found. Upload it to the repo root.")

# PyTorch 2.6+ fix: allow full deserialization
db_data = torch.load(EMBED_FILE, weights_only=False)

# ================= Load multilingual embedding model =================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ================= Pydantic model =================
class QueryRequest(BaseModel):
    message: str

# ================= Semantic search endpoint =================
@app.post("/query")
def query(request: QueryRequest):
    try:
        question_emb = model.encode(request.message, convert_to_tensor=True)
        similarities = [util.cos_sim(question_emb, e["embedding"]).item() for e in db_data]
        best_idx = similarities.index(max(similarities))
        return JSONResponse({"ai_reply": db_data[best_idx]["content"]})
    except Exception as e:
        return JSONResponse({"ai_reply": f"[Error] {e}"})

# ================= Run on Render =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
