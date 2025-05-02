import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from model import TransformerClassifier
from math import log
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dosya yolları
DATA_DIR = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\data"
DATASET_PATH = f"{DATA_DIR}\\encoded_dataset.json"
VOCAB_PATH = f"{DATA_DIR}\\vocab_mapping.json"
LABEL_FREQ_PATH = f"{DATA_DIR}\\label_freq.json"
METRICS_PATH = "rl_metrics_summary.txt"
MODEL_SAVE_PATH = r"C:\\Users\\uzman\\PycharmProjects\\tcdd_proje\\transformer_rl.pt"

# Hiperparametreler
EPISODES = 150
BATCH_SIZE = 32
LR = 3e-5
REWARD_CORRECT = 3.0
REWARD_INCORRECT_BASE = -1.0
SOFTMAX_TEMP = 0.95
PATIENCE = 10
STATION_CAPACITY = 5
MAX_LEN = 128
EMBED_SIZE = 384
NUM_HEADS = 6
NUM_LAYERS = 10
FF_DIM = 768
DROPOUT = 0.15

class FocalLossWithLS(nn.Module):
    def __init__(self, gamma=1.0, smoothing=0.1, weight=None):
        super(FocalLossWithLS, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, input, target):
        log_probs = torch.log_softmax(input, dim=-1)
        probs = torch.softmax(input, dim=-1)

        num_classes = input.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_class_f1_scores():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        f1_scores = {}
        for line in lines:
            if line.strip() and "|" in line:
                parts = line.split()
                label = parts[0].strip().lower()
                score = float(parts[-2])
                f1_scores[label] = score
        return f1_scores
    except:
        return {}

def get_fallback_penalty_classes(threshold=0.25):
    scores = load_class_f1_scores()
    return {label for label, score in scores.items() if score < threshold}

def reinforce_loss(logits, actions, rewards, entropy_beta=0.0):
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()
    loss = -torch.mean(selected_log_probs * rewards) - entropy_beta * entropy
    return loss, selected_log_probs.mean().item(), entropy.item()

def main():
    data = load_data()
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(LABEL_FREQ_PATH, "r", encoding="utf-8") as f:
        label_freq = json.load(f)

    f1_scores = load_class_f1_scores()
    fallback_penalty_classes = get_fallback_penalty_classes()
    SMALL_CLASS_THRESHOLD = 1250
    small_classes = [int(cls) for cls, count in label_freq.items() if count <= SMALL_CLASS_THRESHOLD]

    vocab_size = max(vocab.values()) + 1
    num_classes = max(item["label"] for item in data) + 1
    id2label = {v: k.lower() for k, v in vocab.items() if isinstance(v, int)}

    class_weights = torch.tensor([
        1.0 / log(1.2 + label_freq.get(str(i), 1)) for i in range(num_classes)
    ], dtype=torch.float32).to(device)

    criterion = FocalLossWithLS(gamma=1.2, smoothing=0.1, weight=class_weights)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        num_classes=num_classes,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pooling="mean"
    ).to(device)
    ema_model = TransformerClassifier(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        num_classes=num_classes,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pooling="mean"
    ).to(device)
    ema_model.load_state_dict(model.state_dict())  # başlangıçta aynı
    ema_decay = 0.998 # EMA yavaş decay katsayısı

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                0.5 * (1.0 + np.cos(
                    np.pi * (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))))
            )

        return LambdaLR(optimizer, lr_lambda)

    num_training_steps = EPISODES * (len(data) // BATCH_SIZE)
    T_MAX = num_training_steps * 2  # daha uzun decay süresi
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # örn. %10 warmup
        num_training_steps=num_training_steps
    )
    reward_trend = []
    best_f1_macro = float("-inf")
    no_improve_count = 0
    recent_struggling_classes = set()

    for episode in range(1, EPISODES + 1):
        curriculum_scaler = min(1.0, episode / 30)
        weights = np.array([
            ((1.0 + curriculum_scaler) + curriculum_scaler * (max(1.0, 2.0 - f1_scores.get(str(item["label"]), 0.5)))) *
            (1.0 / (label_freq.get(str(item["label"]), 1) ** 0.5))
            for item in data
        ])

        weights /= weights.sum()
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        sampled_indices = np.random.choice(len(data), size=len(data), replace=False, p=weights)
        data_sampled = [data[i] for i in sampled_indices]
        augment_counter = 0

        station_capacity = {str(i): STATION_CAPACITY for i in range(num_classes)}
        total_loss, correct, total = 0, 0, 0
        logprobs_all, entropy_all, y_true_all, y_pred_all = [], [], [], []
        pred_counter = Counter()
        episode_rewards = []

        data_sampled = [
            item for item in data_sampled
            if
            random.random() < min(1.0, 0.4 + (label_freq.get(str(item["label"]), 1) / max(1, max(label_freq.values()))))
        ]

        hybrid_alpha = max(0.25, 0.6 * (0.98 ** episode))
        explore_ratio = max(0.02, 0.25 - (episode / EPISODES) * 0.2)
        overpredict_penalty = 0.85 + 0.1 * (1 - episode / EPISODES)
        criterion.gamma = max(1.2, 1.8 - (episode / EPISODES) * 0.6)
        # === Gumbel ve Entropy ayarları ===
        gumbel_temp = max(0.5, 1.0 - (episode / EPISODES) * 0.7)  # Gumbel sıcaklığı
        entropy_beta = min(0.05, 0.01 + 0.002 * episode)  # Entropy regularization

        for i in tqdm(range(0, len(data_sampled), BATCH_SIZE), desc=f"Episode {episode}/{EPISODES}"):
            batch = data_sampled[i:i + BATCH_SIZE]
            input_ids = [item["input_ids"] for item in batch]
            labels = [item["label"] for item in batch]

            augment_ratio = min(1.0, (episode / 8) ** 0.7)

            augment_counter = 0
            augmented_input_ids = []
            augmented_labels = []

            for ids, label in zip(input_ids, labels):
                original_ids = ids.copy()
                if label in small_classes:
                    augmented = False

                    # 1. Token Dropout
                    if random.random() < augment_ratio * 0.5 and len(original_ids) > 4:
                        drop_idx = random.choice(range(1, len(original_ids)))
                        del original_ids[drop_idx]
                        augmented = True

                    # 2. Token Swap
                    if random.random() < augment_ratio * 0.5 and len(original_ids) > 3:
                        idx1, idx2 = random.sample(range(1, len(original_ids)), 2)
                        original_ids[idx1], original_ids[idx2] = original_ids[idx2], original_ids[idx1]
                        augmented = True

                    # 3. Token Duplication
                    if random.random() < augment_ratio * 0.3:
                        dup_idx = random.choice(range(len(original_ids)))
                        original_ids.insert(dup_idx, original_ids[dup_idx])
                        augmented = True

                    # 4. Random Noise Token
                    if random.random() < augment_ratio * 0.3:
                        noise_token = random.randint(1, len(vocab))
                        original_ids.append(noise_token)
                        augmented = True
                    # 5. Token Shuffle (shuffle all tokens except [0] padding)
                    if random.random() < augment_ratio * 0.4 and len(original_ids) > 3:
                        shuffle_part = original_ids[1:]  # skip start token if needed
                        random.shuffle(shuffle_part)
                        original_ids = [original_ids[0]] + shuffle_part
                        augmented = True

                    if augmented:
                        augment_counter += 1

                # Pad veya truncate işlemi
                if len(original_ids) > 6:
                    original_ids = original_ids[:6]
                elif len(original_ids) < 6:
                    original_ids += [0] * (6 - len(original_ids))

                augmented_input_ids.append(original_ids)
                augmented_labels.append(label)

            # Tüm input ve label tensor'larını bu listelerden oluştur
            input_tensor = torch.tensor(augmented_input_ids, dtype=torch.long, device=device)
            label_tensor = torch.tensor(augmented_labels, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type="cuda"):
              logits = model(input_tensor) / SOFTMAX_TEMP

            probs = torch.softmax(logits.float(), dim=-1)

            if random.random() < explore_ratio:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs))) * gumbel_temp
                actions = torch.argmax(torch.log(probs + 1e-9) + gumbel_noise, dim=-1)
            else:
                actions = torch.argmax(probs, dim=-1)

            pred_counter.update(actions.tolist())
            y_true_all.extend(label_tensor.tolist())
            y_pred_all.extend(actions.tolist())

            rewards = []
            for j in range(len(batch)):
                true_label = label_tensor[j].item()
                pred_label = actions[j].item()
                label_str = id2label.get(pred_label, "")

                # F1 tabanlı bonus
                f1_bonus = max(1.0, 1.5 - f1_scores.get(id2label.get(true_label, ""), 0.5))

                # frequency scale
                freq = label_freq.get(str(pred_label), 1)
                penalty_scale = min(1.0, 500 / freq)

                # === Reward hesaplama ===
                if pred_label == true_label:
                    reward = REWARD_CORRECT * f1_bonus
                    reward += random.uniform(0.0, 0.5)  # hafif pozitif noise
                    reward += log((1 + episode) / 2) * 0.3
                    if true_label in small_classes:
                        reward += 0.15
                    # struggling class bonusu
                    if str(true_label) in recent_struggling_classes:
                        reward += 0.5
                else:
                    reward = REWARD_INCORRECT_BASE * penalty_scale
                    if true_label in small_classes:
                        reward -= 0.15

                    # az exploration'da hata yapıyorsa daha çok ceza
                    if explore_ratio < 0.1:
                        reward *= 1.25
                    # Top-3 içinde doğru varsa ek ödül
                    topk = torch.topk(probs[j], k=3).indices.tolist()
                    if true_label in topk and pred_label != true_label:
                        reward += 0.75

                    # Diversity boost: üst üste aynı sınıfı tahmin ediyorsa ödülü biraz azalt
                    if j >= 2 and pred_label == actions[j - 1].item() == actions[j - 2].item():
                        reward *= 0.9

                # fallback sınıfıysa daha az ödül
                if label_str in fallback_penalty_classes or pred_label in small_classes:
                    reward *= 0.7

                # overpredict edilen sınıfa ceza (soft scale)
                over_threshold = len(y_pred_all) * 0.15
                if pred_counter[pred_label] > over_threshold:
                    excess = pred_counter[pred_label] - over_threshold
                    if excess > 0:
                        penalty_factor = 1 + np.log1p(excess / over_threshold)
                        reward *= overpredict_penalty * penalty_factor

                # reward sınırları
                reward = max(min(reward, 6.0), -3.0)
                rewards.append(reward)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            episode_rewards.append(rewards.mean().item())
            # Küçük sınıflara özel loss ağırlığı
            class_counts = {}
            for item in data_sampled:
                class_counts[item["label"]] = class_counts.get(item["label"], 0) + 1
            small_classes = [cls for cls, count in class_counts.items() if count < SMALL_CLASS_THRESHOLD]

            label_tensor_weights = torch.ones_like(label_tensor, dtype=torch.float32)
            for idx, lbl in enumerate(label_tensor.tolist()):
                if lbl in small_classes:
                    label_tensor_weights[idx] = 1 + (500 / (label_freq.get(str(lbl),500)))  # küçük sınıf loss ağırlığı 2 kat

            label_tensor_weights = label_tensor_weights.to(device)

            rl_loss, mean_logprob, entropy_val = reinforce_loss(logits.float(), actions, rewards, entropy_beta=entropy_beta)
            ce_loss = (criterion(logits.float(), label_tensor) * label_tensor_weights).mean()
            loss = hybrid_alpha * rl_loss + (1 - hybrid_alpha) * ce_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # EMA ağırlık güncellemesi
            with torch.no_grad():
                for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)


            total_loss += loss.item()
            correct += (actions == label_tensor).sum().item()
            total += len(batch)
            logprobs_all.append(mean_logprob)
            entropy_all.append(entropy_val)
        # Her 20 episode'da learning rate'yi hafif yükselt
        if episode % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.5

        acc = correct / total * 100
        avg_logprob = np.mean(logprobs_all)
        avg_entropy = np.mean(entropy_all)
        avg_reward = np.mean(episode_rewards)
        reward_trend.append(avg_reward)
        if episode % 10 == 0:
            print(f"Episode {episode} - Augment counter in samples: {augment_counter}")

        f1_macro = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        f1_micro = f1_score(y_true_all, y_pred_all, average="micro", zero_division=0)
        # === Performansı düşük sınıfları takip et (adaptive reward için) ===
        class_f1_dict = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        recent_struggling_classes = set()
        for cls_id in range(num_classes):
            label = id2label.get(cls_id, "")
            f1 = class_f1_dict.get(label, {}).get("f1-score", 1.0)
            if f1 < 0.4:
                recent_struggling_classes.add(str(cls_id))

        print(f"Episode {episode} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}% | "
              f"Avg log_prob: {avg_logprob:.4f} | Avg entropy: {avg_entropy:.4f} | Avg reward: {avg_reward:.4f}")
        print(f"F1 Macro: {f1_macro:.4f} | F1 Micro: {f1_micro:.4f}")
        min_delta = 0.001

        if f1_macro > best_f1_macro + min_delta:
            best_f1_macro = f1_macro
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"Early stopping at episode {episode} (no f1 improvement in {PATIENCE} episodes)")
                break

    plt.figure(figsize=(8, 5))
    plt.plot(reward_trend, marker='o')
    plt.title("Reward Trend by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_trend.png")
    plt.close()
    torch.save(ema_model.state_dict(), MODEL_SAVE_PATH)
    print("\n✅ Model saved at:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
