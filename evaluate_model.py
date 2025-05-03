import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score,
    confusion_matrix, classification_report
)
from model import TransformerClassifier

def load_data(data_dir):
    with open(os.path.join(data_dir, "encoded_dataset.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(data_dir, "vocab_mapping.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(data_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(data_dir, "label_freq.json"), "r", encoding="utf-8") as f:
        label_freq = json.load(f)
    return data, vocab, label2id, label_freq

def load_model(model_path, vocab_size, num_classes, config, device):
    model = TransformerClassifier(
        vocab_size=vocab_size,
        max_len=config["max_len"],
        num_classes=num_classes,
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        ff_hidden_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling="mean"
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(model, data, id2label, label_freq, config, save_dir, device, viz_dir):
    num_classes = len(id2label)
    station_capacity = {k: config["station_capacity"] for k in id2label.values()}
    fallback_flags, y_true, y_pred, y_scores = [], [], [], []
    fallback_used = 0
    fallback_log = []
    fallback_steps = []
    entropy_vals= []
    valid_classes = sorted([int(k) for k, v in label_freq.items() if v >= config["min_class_freq"]])

    for i, item in enumerate(data):
        if "label" not in item or item["label"] not in valid_classes:
            continue

        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long).to(device)
        logits = model(input_ids) / config["softmax_temp"]
        probs = torch.softmax(logits, dim=-1).squeeze()
        probs_np = probs.detach().cpu().numpy()
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))
        entropy_vals.append(entropy)
        sorted_indices = torch.argsort(probs, descending=True)

        true_label = item["label"]
        prediction = None
        for idx in sorted_indices:
            label_str = id2label[idx.item()]
            if station_capacity.get(label_str, 0) > 0:
                prediction = idx.item()
                station_capacity[label_str] -= 1
                fallback_flags.append(False)
                break

        if prediction is None:
            prediction = sorted_indices[0].item()
            fallback_used += 1
            fallback_flags.append(True)

        y_true.append(true_label)
        y_pred.append(prediction)
        y_scores.append(probs.detach().cpu().numpy())

        if i % config["capacity_refresh_step"] == 0 and i > 0:
            recent_fallback = fallback_flags[-config["capacity_refresh_step"]:]
            fallback_rate = sum(recent_fallback) / len(recent_fallback)
            fallback_log.append(fallback_rate)
            fallback_steps.append(i)

            pred_labels = [id2label[y] for y in y_pred[-config["capacity_refresh_step"]:] if y in id2label]
            pred_series = pd.Series(pred_labels).value_counts().sort_values(ascending=False)

            plt.figure(figsize=(12, 5))
            sns.barplot(x=pred_series.index, y=pred_series.values)
            plt.xticks(rotation=90)
            plt.title(f"Tahmin Dağılımı (step {i})")
            plt.ylabel("Tahmin Sayısı")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"sim_pred_dist_step_{i}.png"))
            plt.close()

            for k in station_capacity:
                station_capacity[k] = min(config["station_capacity"], station_capacity[k] + config["capacity_recovery"])


    # Metrikler
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    top3_acc = top_k_accuracy_score(y_true, y_scores, k=3)

    print(f"\nTest Tamamlandı")
    print(f"→ Accuracy: {acc:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f} | Top-3 Accuracy: {top3_acc:.4f}")
    print(f"→ Toplam kayıt: {len(y_true)} | Fallback kullanılan: {fallback_used}")

    # Rapor ve grafikler
    os.makedirs(save_dir, exist_ok=True)
    viz_dir = os.path.join(base_dir, "test_values\capacity_refresh_step")
    os.makedirs(viz_dir, exist_ok=True)

    with open(os.path.join(save_dir, "rl_y_true_pred.json"), "w", encoding="utf-8") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2)

    report = classification_report(
        y_true, y_pred,
        labels=list(valid_classes),
        target_names=[id2label[i] for i in valid_classes],
        digits=3,
        zero_division=0
    )
    with open(os.path.join(save_dir, "rl_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f} | Top-3 Accuracy: {top3_acc:.4f}\n\n")
        f.write(report)

    # Görseller
    plot_confusion_matrix(y_true, y_pred, valid_classes, id2label, save_dir)
    plot_fallback_analysis(y_true, fallback_flags, id2label, save_dir)
    plot_station_preference(y_pred, id2label, valid_classes, save_dir)
    report_by_class_size(y_true, y_pred, label_freq, id2label, save_dir, threshold=config["small_class_threshold"])
    analyze_component_station_match(data, y_pred, id2label, vocab, save_dir)
    # Fallback kullanım eğrisi
    if fallback_log:
        plt.figure(figsize=(8, 4))
        plt.plot(fallback_steps, fallback_log, marker='o')
        plt.title("Fallback Kullanım Oranı (Simülasyon Süreci)")
        plt.xlabel("Adım")
        plt.ylabel("Fallback Oranı")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fallback_usage_trend.png"))
        plt.close()

    # Entropi eğrisi
    if entropy_vals:
        plt.figure(figsize=(8, 4))
        plt.plot(entropy_vals, alpha=0.8)
        plt.title("Model Entropi (Belirsizlik) Eğilimi")
        plt.xlabel("Adım")
        plt.ylabel("Entropi")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "entropy_trend.png"))
        plt.close()


def plot_confusion_matrix(y_true, y_pred, valid_classes, id2label, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    cm_df = pd.DataFrame(cm, index=[id2label[i] for i in valid_classes],
                         columns=[id2label[i] for i in valid_classes])
    cm_df.to_csv(os.path.join(save_dir, "rl_confusion_matrix.csv"))
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_df, cmap="Reds")
    plt.title("Confusion Matrix - RL Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rl_confusion_matrix.png"))
    plt.close()

def plot_fallback_analysis(y_true, fallback_flags, id2label, save_dir):
    fallback_labels = [y_true[i] for i, flag in enumerate(fallback_flags) if flag]
    fallback_counts = pd.Series([id2label[l] for l in fallback_labels]).value_counts()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=fallback_counts.index, y=fallback_counts.values)
    plt.xticks(rotation=90)
    plt.title("Fallback ile Tahmin Edilen Gerçek Sınıflar")
    plt.ylabel("Fallback Sayısı")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rl_fallback_analysis.png"))
    plt.close()

def plot_station_preference(y_pred, id2label, valid_classes, save_dir):
    preferred_counts = pd.Series([id2label[p] for p in y_pred if p in valid_classes]).value_counts()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=preferred_counts.index, y=preferred_counts.values)
    plt.xticks(rotation=90)
    plt.title("Tahminlerde Tercih Edilen İstasyonlar")
    plt.ylabel("Tahmin Sayısı")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rl_station_preference.png"))
    plt.close()

def report_by_class_size(y_true, y_pred, label_freq, id2label, save_dir, threshold):
    big_classes = [i for i, count in label_freq.items() if int(count) >= threshold]
    small_classes = [i for i, count in label_freq.items() if int(count) < threshold]
    big_y_true = [yt for yt in y_true if str(yt) in big_classes]
    big_y_pred = [yp for yp, yt in zip(y_pred, y_true) if str(yt) in big_classes]
    small_y_true = [yt for yt in y_true if str(yt) in small_classes]
    small_y_pred = [yp for yp, yt in zip(y_pred, y_true) if str(yt) in small_classes]

    with open(os.path.join(save_dir, "rl_metrics_summary.txt"), "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(" BÜYÜK SINIFLAR METRİKLERİ:\n")
        f.write(classification_report(big_y_true, big_y_pred, digits=3, zero_division=0))
        f.write("\n" + "="*60 + "\n")
        f.write(" KÜÇÜK SINIFLAR METRİKLERİ:\n")
        f.write(classification_report(small_y_true, small_y_pred, digits=3, zero_division=0))
def analyze_component_station_match(data, y_pred, id2label, vocab, save_dir):
    id2token = {v: k for k, v in vocab.items()}

    # Anahtar kelime listesi (komponentlerle eşleşecek kelimeler)
    component_keywords = [
        "tekerlek", "fren", "sabo", "boden", "duvar", "dikme", "kapı", "basamak", "yaprak", "boji", "branda",
        "çatı", "süspansiyon", "el freni", "taban", "dingil", "iskelet", "yük", "menteşe", "havalandırma",
        "helezon", "tampon", "şasi", "yazı", "işaret", "payanda", "bağlantı", "plaka", "donanım", "paketleme"
    ]

    comp_data = []
    for item, pred_label in zip(data, y_pred):
        tokens = [id2token.get(tok, "") for tok in item["input_ids"]]
        text = " ".join(tokens).lower()
        for comp in component_keywords:
            if comp in text:
                comp_data.append((comp, id2label[pred_label]))
                break

    if not comp_data:
        print(" Uyum analizi için yeterli komponent verisi bulunamadı.")
        return

    df = pd.DataFrame(comp_data, columns=["komponent", "tahmin_istasyon"])
    pivot = df.pivot_table(index="komponent", columns="tahmin_istasyon", aggfunc="size", fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct_rounded = pivot_pct.round(1)

    # Sonuçları kaydet
    output_path = os.path.join(save_dir, "komponent_istasyon_uyumu.txt")
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("Komponent - Tahmin Edilen İstasyon Uyumu (%)\n\n")
        f.write(pivot_pct_rounded.to_string(justify="center", float_format="%.1f").expandtabs(4))
    output_path_xlsx = os.path.join(save_dir, "komponent_istasyon_uyumu.xlsx")
    pivot_pct_rounded.to_excel(output_path_xlsx)
    print(f"\n Komponent - Tahmin Edilen İstasyon Uyumu kaydedildi: {output_path_xlsx}")

    print(f"\n Komponent - Tahmin Edilen İstasyon Uyumu kaydedildi: {output_path}")

if __name__ == "__main__":
    config = {
        "max_len": 128,
        "embed_size": 384,
        "num_heads": 6,
        "num_layers": 10,
        "ff_dim": 768,
        "dropout": 0.15,
        "softmax_temp": 1.1,
        "station_capacity": 5,
        "capacity_refresh_step": 3000,
        "capacity_recovery": 1,
        "min_class_freq": 1,
        "small_class_threshold": 1250
    }

    base_dir = r"C:\Users\uzman\PycharmProjects\tcdd_proje"
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(base_dir, "transformer_rl.pt")
    save_dir = os.path.join(base_dir, "test_values")
    viz_dir = os.path.join(base_dir, "test_values\capacity_refresh_step")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, vocab, label2id, label_freq = load_data(data_dir)
    id2label = {v: k for k, v in label2id.items()}
    model = load_model(model_path, max(vocab.values()) + 1, len(label2id), config, device)
    evaluate(model, data, id2label, label_freq, config, save_dir, device,viz_dir)

