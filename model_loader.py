import torch
import os
import json
from model import TransformerClassifier

def load_model_and_vocab(model_path, data_dir, config, device):
    with open(os.path.join(data_dir, "vocab_mapping.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(os.path.join(data_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}

    model = TransformerClassifier(
        vocab_size=max(vocab.values()) + 1,
        max_len=config["max_len"],
        num_classes=len(label2id),
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        ff_hidden_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling="mean"
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vocab, id2label
