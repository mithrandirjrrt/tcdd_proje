import torch
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, top_k_accuracy_score
from model import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dosya yolları
DATA_DIR = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\data"
SAVE_DIR = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\test_values"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_PATH = f"{DATA_DIR}\\encoded_dataset.json"
VOCAB_PATH = f"{DATA_DIR}\\vocab_mapping.json"
LABEL2ID_PATH = f"{DATA_DIR}\\label2id.json"
LABEL_FREQ_PATH = f"{DATA_DIR}\\label_freq.json"
MODEL_PATH = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\transformer_rl.pt"

# Parametreler
MAX_LEN = 128
EMBED_SIZE = 384
NUM_HEADS = 6
NUM_LAYERS = 10
FF_DIM = 768
DROPOUT = 0.15
SOFTMAX_TEMP = 1.1
STATION_CAPACITY = 5
CAPACITY_REFRESH_STEP = 3000
CAPACITY_RECOVERY = 1
MIN_TEST_CLASS_FREQ = 1
SMALL_CLASS_THRESHOLD = 1250

# Veri yükle
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open(LABEL2ID_PATH, "r", encoding="utf-8") as f:
    label2id = json.load(f)
with open(LABEL_FREQ_PATH, "r", encoding="utf-8") as f:
    label_freq = json.load(f)

id2label = {v: k for k, v in label2id.items()}
num_classes = len(label2id)

# Eksik sınıflar loglama
seen_labels = set([item["label"] for item in data if "label" in item])
defined_labels = set(int(k) for k in label_freq.keys())
missing = seen_labels - defined_labels
if missing:
    print("⚠️ WARNING: These labels are in data but not in label_freq.json:", missing)

valid_classes = {int(cls) for cls, count in label_freq.items() if count >= MIN_TEST_CLASS_FREQ}
valid_class_list = sorted(valid_classes)

# Model
model = TransformerClassifier(
    vocab_size=max(vocab.values()) + 1,
    max_len=MAX_LEN,
    num_classes=num_classes,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    ff_hidden_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pooling="mean"
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Tahmin kayıtları
station_capacity = {label: STATION_CAPACITY for label in label2id.keys()}
y_true, y_pred, y_scores, fallback_flags = [], [], [], []
fallback_used = 0

with torch.no_grad():
    for i, item in enumerate(data):
        if "label" not in item or item["label"] is None:
            continue
        if item["label"] not in valid_classes:
            continue

        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long).to(device)
        true_label = item["label"]
        logits = model(input_ids) / SOFTMAX_TEMP
        probs = torch.softmax(logits, dim=-1).squeeze()
        sorted_indices = torch.argsort(probs, descending=True)

        final_prediction = -1
        for idx in sorted_indices:
            label_str = id2label[idx.item()]
            if station_capacity.get(label_str, 0) > 0:
                final_prediction = idx.item()
                station_capacity[label_str] -= 1
                fallback_flags.append(False)
                break
        else:
            fallback_used += 1
            final_prediction = sorted_indices[0].item()
            fallback_flags.append(True)

        y_true.append(true_label)
        y_pred.append(final_prediction)
        y_scores.append(probs.cpu().numpy())

        if i % CAPACITY_REFRESH_STEP == 0 and i > 0:
            for k in station_capacity:
                station_capacity[k] = min(STATION_CAPACITY, station_capacity[k] + CAPACITY_RECOVERY)

# METRİKLER
acc = accuracy_score(y_true, y_pred)
f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
top3_acc = top_k_accuracy_score(y_true, y_scores, k=3, labels=list(range(num_classes)))

print(f"\n✅ Test tamamlandı.")
print(f"→ Accuracy: {acc:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f} | Top-3 Accuracy: {top3_acc:.4f}")
print(f"→ Toplam kayıt: {len(y_true)} | Fallback kullanılan tahmin sayısı: {fallback_used}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=valid_class_list)
cm_df = pd.DataFrame(cm, index=[id2label[i] for i in valid_class_list],
                     columns=[id2label[i] for i in valid_class_list])
cm_df.to_csv(os.path.join(SAVE_DIR, "rl_confusion_matrix.csv"), encoding="utf-8-sig")
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, cmap="Reds", xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix - RL Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rl_confusion_matrix.png"))
plt.close()

# Tahmin Tercih Grafiği
preferred_counts = pd.Series([id2label[p] for p in y_pred if p in valid_classes]).value_counts().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
sns.barplot(x=preferred_counts.index, y=preferred_counts.values)
plt.xticks(rotation=90)
plt.title("Tahminlerde Tercih Edilen İstasyonlar")
plt.xlabel("İstasyon (Sınıf)")
plt.ylabel("Tahmin Sayısı")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rl_station_preference.png"))
plt.close()

# Fallback ile Tahmin Edilen Gerçekler
fallback_labels = [y_true[i] for i in range(len(y_true)) if fallback_flags[i]]
fallback_counts = pd.Series([id2label[l] for l in fallback_labels]).value_counts().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
sns.barplot(x=fallback_counts.index, y=fallback_counts.values)
plt.xticks(rotation=90)
plt.title("Fallback ile Tahmin Edilen Gerçek Sınıflar")
plt.xlabel("Gerçek Sınıf")
plt.ylabel("Fallback Sayısı")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rl_fallback_analysis.png"))
plt.close()

# Classification Report
report = classification_report(
    y_true, y_pred,
    target_names=[id2label[i] for i in valid_class_list],
    labels=valid_class_list,
    digits=3,
    zero_division=0
)
with open(os.path.join(SAVE_DIR, "rl_metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\nF1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f} | Top-3 Accuracy: {top3_acc:.4f}\n\n")
    f.write(report)

# Küçük ve büyük sınıflara göre ayır
big_classes = [i for i in valid_class_list if label_freq.get(str(i), 0) >= SMALL_CLASS_THRESHOLD]
small_classes = [i for i in valid_class_list if label_freq.get(str(i), 0) < SMALL_CLASS_THRESHOLD]

big_y_true = [yt for yt, yt_class in zip(y_true, y_true) if yt_class in big_classes]
big_y_pred = [yp for yp, yt_class in zip(y_pred, y_true) if yt_class in big_classes]
small_y_true = [yt for yt, yt_class in zip(y_true, y_true) if yt_class in small_classes]
small_y_pred = [yp for yp, yt_class in zip(y_pred, y_true) if yt_class in small_classes]

if big_classes:
    report_big = classification_report(
        big_y_true, big_y_pred,
        target_names=[id2label[i] for i in big_classes],
        labels=big_classes,
        digits=3,
        zero_division=0
    )
else:
    report_big = "BÜYÜK sınıflar için yeterli veri bulunamadı.\n"

if small_classes:
    report_small = classification_report(
        small_y_true, small_y_pred,
        target_names=[id2label[i] for i in small_classes],
        labels=small_classes,
        digits=3,
        zero_division=0
    )
else:
    report_small = "KÜÇÜK sınıflar için yeterli veri bulunamadı.\n"

with open(os.path.join(SAVE_DIR, "rl_metrics_summary.txt"), "a", encoding="utf-8") as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"🔎 BÜYÜK SINIFLAR (>= {SMALL_CLASS_THRESHOLD}) METRİKLERİ:\n")
    f.write(report_big)
    f.write("\n" + "="*60 + "\n")
    f.write(f"🔍 KÜÇÜK SINIFLAR (< {SMALL_CLASS_THRESHOLD}) METRİKLERİ:\n")
    f.write(report_small)

with open(os.path.join(SAVE_DIR, "rl_y_true_pred.json"), "w", encoding="utf-8") as f:
    json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2)

print("📁 Tüm çıktılar ve görseller kaydedildi:", SAVE_DIR)
