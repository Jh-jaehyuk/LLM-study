import torch
import torch.nn as nn
import math

torch.set_default_device("mps")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len=64, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.segment = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embedding_dim, max_len=seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
