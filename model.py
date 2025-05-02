import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_size]
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, E = x.size()
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out, _ = self.attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.dropout(self.fc_out(out))


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class EncoderStack(nn.Module):
    def __init__(self, num_layers, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        num_classes,
        embed_size=384,
        num_heads=6,
        ff_hidden_dim=1024,
        num_layers=8,
        dropout=0.1,
        pooling="mean"  # ya "mean" ya da "cls"
    ):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.encoder_stack = EncoderStack(num_layers, embed_size, num_heads, ff_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling.lower()
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder_stack(x)

        if self.pooling == "cls":
            out = x[:, 0, :]  # [CLS] token (ilk token) üzerinden sınıflandırma
        else:
            out = x.mean(dim=1)  # Mean pooling

        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
