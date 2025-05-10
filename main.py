from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from model_loader import load_model_and_vocab
from threading import Lock
import math
import random
import json
import os
import shutil

class PredictRequest(BaseModel):
    vagon_no: str
    vagon_tipi: str
    komponent: str

class ConfirmRequest(BaseModel):
    vagon_no: str

class CompleteRequest(BaseModel):
    vagon_no: str
    vagon_tipi: str
    komponent: str
    neden: str
    istasyon: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {
    "max_len": 128,
    "embed_size": 384,
    "num_heads": 6,
    "num_layers": 10,
    "ff_dim": 768,
    "dropout": 0.15,
    "softmax_temp": 1.1
}

base_dir = "."
data_dir = f"{base_dir}/data"
model_path = f"{base_dir}/transformer_rl.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vocab, id2label = load_model_and_vocab(model_path, data_dir, config, device)

station_capacity = {ist: 5 for ist in id2label.values()}
capacity_lock = Lock()
pending_predictions = []

ACTIVE_REPAIRS_FILE = "aktif_bakimlar.json"
COMPLETED_REPAIRS_FILE = "tamamlanan_bakimlar.json"

if not os.path.exists(ACTIVE_REPAIRS_FILE):
    with open(ACTIVE_REPAIRS_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(COMPLETED_REPAIRS_FILE):
    with open(COMPLETED_REPAIRS_FILE, "w") as f:
        json.dump([], f)

def save_backup():
    try:
        shutil.copy(ACTIVE_REPAIRS_FILE, "aktif_bakimlar_backup.json")
    except Exception as e:
        print(f"Yedekleme hatası: {e}")

@app.get("/active_repairs")
def get_active_repairs():
    if not os.path.exists(ACTIVE_REPAIRS_FILE):
        with open(ACTIVE_REPAIRS_FILE, "w") as f:
            json.dump({}, f)
    with open(ACTIVE_REPAIRS_FILE) as f:
        return json.load(f)

@app.post("/activate_repair")
def activate_repair(req: PredictRequest):
    input_ids = [
        vocab.get("__cls__", 0),
        vocab.get(req.vagon_no, 0),
        vocab.get(req.vagon_tipi, 0),
        vocab.get(req.komponent, 0)
    ]
    input_tensor = torch.tensor([input_ids + [0] * (config["max_len"] - len(input_ids))], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor) / config["softmax_temp"]
        probs = torch.softmax(logits, dim=-1).squeeze()
        top_indices = torch.argsort(probs, descending=True).tolist()

    with open(ACTIVE_REPAIRS_FILE, "r+") as f:
        data = json.load(f)

        already_exists = any(
            r["vagon_no"] == req.vagon_no
            for repairs in data.values()
            for r in repairs
        )
        if already_exists:
            return {"message": "Zaten aktif bakımda."}

        # İstasyon doluluk kontrolü
        istasyon_durum = {ist: len(data.get(ist, [])) for ist in id2label.values()}

        selected = None
        fallback = None

        for idx in top_indices:
            label = id2label[idx]
            if istasyon_durum.get(label, 0) < station_capacity.get(label, 5):
                selected = label
                break

        if not selected:
            selected = id2label[top_indices[0]]
            fallback = selected

        neden = f"{req.komponent} arızası nedeniyle tamire alındı."
        repair = {
            "vagon_no": req.vagon_no,
            "vagon_tipi": req.vagon_tipi,
            "komponent": req.komponent,
            "prediction": selected,
            "confidence": round(probs[top_indices[0]].item(), 4),
            "neden": neden
        }

        if fallback and fallback != selected:
            repair["replaced"] = True
            repair["fallback"] = fallback

        if selected not in data:
            data[selected] = []
        data[selected].append(repair)

        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    save_backup()
    return {"message": "Aktif bakım eklendi.", "prediction": selected, "fallback": fallback if fallback else None}

@app.post("/complete_repair")
def complete_repair(req: CompleteRequest):
    with open(ACTIVE_REPAIRS_FILE, "r+") as f:
        data = json.load(f)
        if req.istasyon in data:
            data[req.istasyon] = [r for r in data[req.istasyon] if r["vagon_no"] != req.vagon_no]
            if not data[req.istasyon]:
                del data[req.istasyon]
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    completed = {
        "vagon_no": req.vagon_no,
        "vagon_tipi": req.vagon_tipi,
        "komponent": req.komponent,
        "neden": req.neden,
        "istasyon": req.istasyon
    }
    with open(COMPLETED_REPAIRS_FILE, "r+") as f:
        tamamlanan = json.load(f)
        tamamlanan.append(completed)
        f.seek(0)
        json.dump(tamamlanan, f, indent=2)
        f.truncate()
    save_backup()
    return {"message": "Bakım tamamlandı."}

@app.get("/")
def root():
    return {"message": "Model API aktif 🎯"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
