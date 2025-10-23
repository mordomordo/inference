# app.py - Inference server
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

# ================= FastAPI =================
app = FastAPI(title="Smart Home Inference Server")

# ================= Load precomputed embeddings =================
# Assume you have saved precomputed embeddings as a .pt file:
# torch.save(db_data, "precomputed_embeddings.pt")
db_data = torch.load("precomputed_embeddings.pt")

# ================= Load embedding model =================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ================= Pydantic model =================
class QueryRequest(BaseModel):
    message: str

# ================= Semantic search endpoint =================
@app.post("/query")
def query(request: QueryRequest):
    try:
        # Encode the user question
        question_emb = model.encode(request.message, convert_to_tensor=True)
        # Compute cosine similarity with all precomputed embeddings
        similarities = [util.cos_sim(question_emb, e["embedding"]).item() for e in db_data]
        # Find the best match
        best_idx = similarities.index(max(similarities))
        answer = db_data[best_idx]["content"]
        return JSONResponse({"ai_reply": answer})
    except Exception as e:
        return JSONResponse({"ai_reply": f"[Error] {e}"})

# ================= Run locally =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
