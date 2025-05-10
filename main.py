from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import torch
from model_loader import load_model_and_vocab
from threading import Lock
import math
import random
import json
import os
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictRequest(BaseModel):
    vagon_no: str
    vagon_tipi: str
    komponent: str
    activation_time: Union[str, None] = None

class ConfirmRequest(BaseModel):
    vagon_no: str

class CompleteRequest(BaseModel):
    vagon_no: str
    vagon_tipi: str
    komponent: str
    neden: str
    istasyon: str
    completion_time: Union[str, None] = None
    activation_time: Union[str, None] = None

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

def encode_input(data: PredictRequest, vocab, max_len=128):
    input_ids = [
        vocab.get("__cls__", 0),
        vocab.get(data.vagon_no, 0),
        vocab.get(data.vagon_tipi, 0),
        vocab.get(data.komponent, 0)
    ]
    return input_ids + [0] * (max_len - len(input_ids))

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Önce vagonun aktif bakımda olup olmadığını kontrol et
        with open(ACTIVE_REPAIRS_FILE, "r") as f:
            active_data = json.load(f)
            for station_repairs in active_data.values():
                for repair in station_repairs:
                    if repair["vagon_no"] == req.vagon_no:
                        logging.warning(f"Vagon {req.vagon_no} zaten aktif bakımda.")
                        raise ValueError("Bu vagon zaten aktif bir bakımda bulunmaktadır. Lütfen başka bir vagon seçiniz.")

        input_ids = encode_input(req, vocab, config["max_len"])
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_tensor) / config["softmax_temp"]
            probs = torch.softmax(logits, dim=-1).squeeze()
            top_indices = torch.argsort(probs, descending=True).tolist()

        # Aktif bakımları kontrol et
        with open(ACTIVE_REPAIRS_FILE, "r") as f:
            data = json.load(f)

        # En uygun istasyonu bul
        assigned = None
        confidence = None
        preferred_station = id2label[top_indices[0]]
        preferred_confidence = probs[top_indices[0]].item()

        for idx in top_indices:
            station = id2label[idx]
            if station not in data or len(data[station]) < 5:
                assigned = station
                confidence = probs[idx].item()
                break

        if assigned is None:
            assigned = preferred_station
            confidence = preferred_confidence

        component_priority_map = {
            "Tekerleğin Bandaj Kısmı": 5,
            "Fren Pnömatik Kısım": 4,
            "Sabo": 3,
            "Boden": 2,
            "Kapı ve Sürme Duvar": 5,
            "Dikme (Platform Vagon)": 2,
            "Yan veya Alın Duvar (Açık Vagon)": 2,
            "Fren Mekanik Kısım": 4,
            "Duvar": 2,
            "Yarı Otomatik Koşum Takımı": 3,
            "Yan Duvar Kapağı (Platform Vagon)": 3,
            "Basamak/Tutamak/Merdiven/Geçit/Korkuluk/Yazı Levhaları vb. Değişik Parçalar": 2,
            "Yaprak Susta": 4,
            "Boji Yan Yastık ve Sustası": 4,
            "Branda Kilitleme Tertibatı (Rils vb) (Özel Tertibatlı Vagon)": 2,
            "Çatı ve Su Sızdırmazlığı (Kapalı Vagon)": 2,
            "Dikme Desteği (Platform Vagon)": 2,
            "Y 25 Bojinin Süspansiyon Sistemi": 5,
            "Topraklama Kablosu": 3,
            "Süspansiyon Bağlantıları": 4,
            "El Freni": 4,
            "Taban": 3,
            "Dingil Kutusu": 5,
            "Vagon Gövdesi İskeleti": 4,
            "Yükün Dağılımı": 2,
            "Kapama Tertibatı/Tespit Sportu (Kapalı Vagon)": 3,
            "ACTS Konteyner Vagonu (Özel Tertibatlı Vagon)": 3,
            "Menteşe/Pim/Sabitleme Civatası (Platform Vagon)": 3,
            "Havalandırma Kapağı (Kapalı Vagon)": 2,
            "Helezon Susta": 4,
            "Dingilli Vagonlarda Süspansiyon Sportu": 4,
            "Monoblok Tekerlek": 5,
            "Alın Kapakların Kapatılması ve Çalıştırılması Tertibatı (Açık Vagon)": 3,
            "Boşi Şasisi": 3,
            "Tampon Plakası": 3,
            "Vagon Şasesi": 4,
            "Vagon Üzerindeki Yazı ve İşaretler": 2,
            "Konteyner Vagonları Üzerindeki Yük Ünitelerinin Emniyete Alınmasına Yönelik Tertibat": 2,
            "Payanda, Kalas, Gerdirme, Bağlantı Tertibatı": 2,
            "Özellikle Yatay veya Düşey Aktarım için Kullanılan Özel Ekipman": 2,
            "Doğrudan veya Dolaylı Bağlantı": 3,
            "Dingil": 5,
            "Bandajlı Tekerlek": 5,
            "Dingil Çatalı Aşınma Plakası": 3,
            "Dikme": 2,
            "Vagon Duvarı veya Kenarı": 3,
            "Tekerlek Gövdesi": 5,
            "İşletme Bozuklukları": 4,
            "Vagon Gövdesi İç Donanımı": 3,
            "Y Bojide Manganlı Aşınma Plakası": 4,
            "Dingil Çatalı Bağlantı Pernosu": 3,
            "Paketleme, Yük Bağlama.": 2,
        }

        component_level = component_priority_map.get(req.komponent, 3)
        label_level = random.randint(1, 5)  # Rastgele öncelik seviyesi

        priority = math.ceil(math.sqrt(component_level * label_level))
        priority = max(1, min(5, priority))

        neden = f"{req.komponent} arızası nedeniyle tamire alındı."

        logging.info(f"Tahmin yapıldı: {req.vagon_no} için {assigned} istasyonuna atandı.")
        return {
            "vagon_no": req.vagon_no,
            "vagon_tipi": req.vagon_tipi,
            "komponent": req.komponent,
            "prediction": assigned,
            "confidence": round(confidence, 4),
            "priority": priority,
            "neden": neden,
            "preferred_station": preferred_station if assigned != preferred_station else None,
            "preferred_confidence": round(preferred_confidence, 4) if assigned != preferred_station else None
        }
    except ValueError as ve:
        logging.error(str(ve))
        raise
    except Exception as e:
        logging.error(f"Tahmin sırasında hata oluştu: {e}")
        raise

@app.post("/activate_repair")
def activate_repair(req: PredictRequest):
    try:
        input_ids = encode_input(req, vocab, config["max_len"])
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_tensor) / config["softmax_temp"]
            probs = torch.softmax(logits, dim=-1).squeeze()
            top_indices = torch.argsort(probs, descending=True).tolist()

        with open(ACTIVE_REPAIRS_FILE, "r+") as f:
            data = json.load(f)

            # Vagon zaten aktif bakımda mı kontrol et
            already_exists = any(
                r["vagon_no"] == req.vagon_no
                for repairs in data.values()
                for r in repairs
            )
            if already_exists:
                logging.warning(f"Vagon {req.vagon_no} zaten aktif bakımda.")
                return {"message": "Zaten aktif bakımda."}

            # En uygun istasyonu bul
            assigned = None
            confidence = None
            preferred_station = id2label[top_indices[0]]
            preferred_confidence = probs[top_indices[0]].item()

            for idx in top_indices:
                station = id2label[idx]
                if station not in data or len(data[station]) < 5:
                    assigned = station
                    confidence = probs[idx].item()
                    break

            if assigned is None:
                logging.warning("Tüm istasyonlar dolu.")
                return {"message": "Tüm istasyonlar dolu. Lütfen daha sonra tekrar deneyin."}

            if assigned not in data:
                data[assigned] = []
            data[assigned].append({
                "vagon_no": req.vagon_no,
                "vagon_tipi": req.vagon_tipi,
                "komponent": req.komponent,
                "prediction": assigned,
                "confidence": round(confidence, 4),
                "neden": f"{req.komponent} arızası nedeniyle tamire alındı.",
                "activation_time": req.activation_time,
                "preferred_station": preferred_station if assigned != preferred_station else None,
                "preferred_confidence": round(preferred_confidence, 4) if assigned != preferred_station else None
            })

            f.seek(0)
            json.dump(data, f, indent=2)
            logging.info(f"Bakım aktivasyonu yapıldı: {req.vagon_no} için {assigned} istasyonuna atandı.")
            return {
                "message": "Bakım başarıyla aktive edildi.",
                "assigned_station": assigned,
                "preferred_station": preferred_station if assigned != preferred_station else None
            }
    except Exception as e:
        logging.error(f"Bakım aktivasyonu sırasında hata oluştu: {e}")
        raise

@app.get("/active_repairs")
def get_active_repairs():
    with open(ACTIVE_REPAIRS_FILE) as f:
        return json.load(f)

@app.get("/history")
def get_history(vagon_no: str):
    try:
        with open(COMPLETED_REPAIRS_FILE, "r") as f:
            data = json.load(f)
        return [item for item in data if item["vagon_no"] == vagon_no]
    except Exception as e:
        logging.error(f"Geçmiş bakım verisi alınamadı: {e}")
        return []

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
        "istasyon": req.istasyon,
        "completion_time": req.completion_time,
        "activation_time": req.activation_time
    }

    with open(COMPLETED_REPAIRS_FILE, "r+") as f:
        tamamlanan = json.load(f)
        tamamlanan.append(completed)
        f.seek(0)
        json.dump(tamamlanan, f, indent=2)
        f.truncate()

    return {"message": "Bakım tamamlandı."}

@app.post("/reset_capacity")
def reset_capacity():
    global station_capacity
    with capacity_lock:
        station_capacity = {ist: 5 for ist in id2label.values()}
    return {"message": "Kapasiteler sıfırlandı 🎯"}

@app.get("/capacities")
def get_capacities():
    return station_capacity

@app.get("/")
def root():
    return {"message": "Model API aktif 🎯"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
