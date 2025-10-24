# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import os

# ================= FastAPI =================
app = FastAPI(title="Smart Home Inference Server")

# ================= Load database =================
DB_FILE = os.path.join(os.path.dirname(__file__), "database.json")
with open(DB_FILE, "r", encoding="utf-8") as f:
    db_data = json.load(f)

# ================= Load model =================
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

# Precompute embeddings if not already
for entry in db_data:
    if "embedding" not in entry:
        entry["embedding"] = model.encode(entry["content"], convert_to_tensor=True)

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

# ================= Run locally =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
