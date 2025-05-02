import pandas as pd
import json
import os
from collections import Counter

# Dosya yolları
DATA_PATH = r"C:\\Users\\uzman\\Desktop\\proje\\Vagon_cleaned.xlsx"
VOCAB_SOURCE_PATH = r"C:\\Users\\uzman\\Desktop\\proje\\Vagon_analysis_output.xlsx"
SAVE_DIR = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Çıktı dosyaları
DATASET_OUT = os.path.join(SAVE_DIR, "encoded_dataset.json")
VOCAB_OUT = os.path.join(SAVE_DIR, "vocab_mapping.json")
LABEL2ID_OUT = os.path.join(SAVE_DIR, "label2id.json")
LABEL_FREQ_OUT = os.path.join(SAVE_DIR, "label_freq.json")
SHEET_STATUS_OUT = os.path.join(SAVE_DIR, "sheet_status.json")
LABEL_PRIORITY_OUT = os.path.join(SAVE_DIR, "label_priority_map.json")
# Komponent öncelikleri (1 = düşük, 5 = yüksek)
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

def build_vocab(excel_path):
    vocab = {"__cls__": 1}
    current_id = 2
    xls = pd.ExcelFile(excel_path)
    sheet_status = {}

    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            if df.empty or df.shape[1] < 2:
                sheet_status[sheet] = "Boş veya yetersiz sütun"
                continue
            values = df.iloc[:, 0].astype(str).tolist()
            for val in values:
                val = val.strip()
                if val not in vocab:
                    vocab[val] = current_id
                    current_id += 1
            sheet_status[sheet] = "✓ Var"
        except Exception as e:
            sheet_status[sheet] = f"Hata: {str(e)}"

    for i in range(1, 6):
        token = f"priority_{i}"
        vocab[token] = current_id
        current_id += 1

    return vocab, sheet_status

def encode_dataset(data_path, vocab):
    df = pd.read_excel(data_path)
    samples = []
    label_counter = Counter()

    for _, row in df.iterrows():
        repeat = int(row["Repeat_Count"])
        vagon_no = str(row["Vagon Numarası"]).strip()
        vagon_tip = str(row["Vagon Tipi"]).strip()
        komponent = str(row["Arıza Görülen Komponentin Adı"]).strip()
        neden = str(row["Tamire Tutulma Nedeni"]).strip()
        hedef = str(row["Tamire Tutan Yer"]).strip()

        if not hedef or hedef.strip().lower() in ["nan", ""]:
            continue

        label_counter[hedef] += repeat

    label2id = {label: idx for idx, label in enumerate(sorted(label_counter))}
    label_freq_by_id = {str(label2id[k]): v for k, v in label_counter.items()}

    for _, row in df.iterrows():
        repeat = int(row["Repeat_Count"])
        vagon_no = str(row["Vagon Numarası"]).strip()
        vagon_tip = str(row["Vagon Tipi"]).strip()
        komponent = str(row["Arıza Görülen Komponentin Adı"]).strip()
        neden = str(row["Tamire Tutulma Nedeni"]).strip()
        hedef = str(row["Tamire Tutan Yer"]).strip()

        if not hedef or hedef.strip().lower() in ["nan", ""]:
            continue

        # Frekansa göre öncelik
        freq = label_counter[hedef]
        if freq > 10000:
            label_priority = 5
        elif freq > 2500:
            label_priority = 4
        elif freq > 1500:
            label_priority = 3
        elif freq > 500:
            label_priority = 2
        else:
            label_priority = 1

        component_priority = component_priority_map.get(komponent, 3)
        combined_priority = max(1, min(5, int((label_priority * component_priority) ** 0.5)))  # dengeli birleştirme

        priority_token = f"priority_{combined_priority}"
        input_tokens = [
            vocab.get("__cls__", 0),
            vocab.get(vagon_no, 0),
            vocab.get(vagon_tip, 0),
            vocab.get(komponent, 0),
            vocab.get(neden, 0),
            vocab.get(priority_token, 0)
        ]

        for _ in range(repeat):
            samples.append({
                "input_ids": input_tokens,
                "label_raw": hedef
            })

    for s in samples:
        s["label"] = label2id[s.pop("label_raw")]
    encoded_filtered = samples  # tüm örnekleri kullan

    return encoded_filtered, label2id, label_freq_by_id

if __name__ == "__main__":
    vocab, sheet_status = build_vocab(VOCAB_SOURCE_PATH)
    encoded, label2id, label_freq = encode_dataset(DATA_PATH, vocab)

    with open(LABEL_PRIORITY_OUT, "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in label_freq.items()}, f, indent=2)
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        json.dump(encoded, f, indent=2, ensure_ascii=False)
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    with open(LABEL2ID_OUT, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    with open(LABEL_FREQ_OUT, "w", encoding="utf-8") as f:
        json.dump(label_freq, f, indent=2)
    with open(SHEET_STATUS_OUT, "w", encoding="utf-8") as f:
        json.dump(sheet_status, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset hazırlandı. Vocab size: {len(vocab)} | Etiket sayısı: {len(label2id)} | Örnek: {len(encoded)}")
