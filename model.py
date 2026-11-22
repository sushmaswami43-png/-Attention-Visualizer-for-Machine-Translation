"""Bi-GRU encoder + multi-head attention GRU decoder (input feeding)."""
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Decoded(NamedTuple):
    """Greedy decoding result. topk_* hold the k best candidates the model considered at every step."""
    ids: List[List[int]]          # chosen token ids per sentence (incl. <eos>)
    attention: torch.Tensor       # [B, T, heads, S]
    topk_ids: torch.Tensor        # [B, T, k]
    topk_probs: torch.Tensor      # [B, T, k]


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_idx: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.memory_proj = nn.Linear(2 * hid_dim, hid_dim)
        self.init_proj = nn.Linear(2 * hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns memory [B, S, H] and initial decoder state [B, H]."""
        packed = pack_padded_sequence(self.dropout(self.embedding(src)), src_len.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, h_n = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        memory = torch.tanh(self.memory_proj(out))
        h0 = torch.tanh(self.init_proj(torch.cat([h_n[0], h_n[1]], dim=-1)))
        return memory, h0


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention of one decoder query over encoder memory."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, "hidden size must be divisible by number of heads"
        self.n_heads, self.d_k = n_heads, dim // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

    def project_memory(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keys/values are computed once per sentence, not once per decoding step."""
        b, s, _ = memory.shape
        split = lambda x: x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, S, d_k]
        return split(self.k_proj(memory)), split(self.v_proj(memory))

    def forward(self, query, keys, values, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """query [B, H]; mask [B, S] (True = real token). Returns context [B, H], weights [B, h, S]."""
        b = query.size(0)
        q = self.q_proj(query).view(b, self.n_heads, 1, self.d_k)
        scores = (q @ keys.transpose(-1, -2)).squeeze(2) / math.sqrt(self.d_k)
        weights = F.softmax(scores.masked_fill(~mask[:, None, :], float("-inf")), dim=-1)
        context = (weights.unsqueeze(2) @ values).squeeze(2).reshape(b, -1)
        return self.out_proj(context), weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_pad: int, tgt_pad: int,
                 tgt_sos: int, tgt_eos: int, emb_dim: int = 192, hid_dim: int = 256,
                 n_heads: int = 4, dropout: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.config: Dict = dict(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                 src_pad=src_pad, tgt_pad=tgt_pad, tgt_sos=tgt_sos, tgt_eos=tgt_eos,
                                 emb_dim=emb_dim, hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,
                                 use_attention=use_attention)
        self.src_pad, self.sos, self.eos, self.n_heads = src_pad, tgt_sos, tgt_eos, n_heads
        self.use_attention = use_attention   # False -> classic seq2seq baseline (no attention)
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, src_pad, dropout)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=tgt_pad)
        self.attention = MultiHeadAttention(hid_dim, n_heads) if use_attention else None
        self.cell = nn.GRUCell(emb_dim + hid_dim, hid_dim)   # input feeding: previous context joins input
        self.combine = nn.Linear(2 * hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _step(self, prev_tok, hidden, prev_ctx, keys, values, mask):
        x = torch.cat([self.dropout(self.tgt_embedding(prev_tok)), prev_ctx], dim=-1)
        hidden = self.cell(x, hidden)
        if self.use_attention:
            ctx, weights = self.attention(hidden, keys, values, mask)
        else:  # baseline: the decoder only sees the encoder's final state
            ctx, weights = torch.zeros_like(hidden), hidden.new_zeros(hidden.size(0), self.n_heads, mask.size(1))
        feat = self.dropout(torch.tanh(self.combine(torch.cat([hidden, ctx], dim=-1))))
        return feat, hidden, ctx, weights

    def _prepare(self, src, src_len):
        memory, hidden = self.encoder(src, src_len)
        keys, values = self.attention.project_memory(memory) if self.use_attention else (None, None)
        return hidden, torch.zeros_like(hidden), keys, values, src != self.src_pad

    def forward(self, src, src_len, tgt_in) -> torch.Tensor:
        """Teacher-forced pass. tgt_in [B, T] -> logits [B, T, V]."""
        hidden, ctx, keys, values, mask = self._prepare(src, src_len)
        feats = []
        for t in range(tgt_in.size(1)):
            feat, hidden, ctx, _ = self._step(tgt_in[:, t], hidden, ctx, keys, values, mask)
            feats.append(feat)
        return self.output(torch.stack(feats, dim=1))   # one big matmul instead of T small ones

    @torch.no_grad()
    def translate(self, src, src_len, max_len: int = 50, topk: int = 5) -> Decoded:
        """Greedy decoding that also records attention and the top-k candidates at every step."""
        self.eval()
        hidden, ctx, keys, values, mask = self._prepare(src, src_len)
        tok = torch.full((src.size(0),), self.sos, dtype=torch.long, device=src.device)
        ids, attn, top_p, top_i = [], [], [], []
        done = torch.zeros_like(tok, dtype=torch.bool)
        for _ in range(max_len):
            feat, hidden, ctx, weights = self._step(tok, hidden, ctx, keys, values, mask)
            probs = F.softmax(self.output(feat), dim=-1)
            p, i = probs.topk(topk, dim=-1)
            tok = i[:, 0]
            ids.append(tok); attn.append(weights); top_p.append(p); top_i.append(i)
            done |= tok == self.eos
            if done.all():
                break
        ids = torch.stack(ids, dim=1).tolist()
        trimmed = [row[:row.index(self.eos) + 1] if self.eos in row else row for row in ids]
        return Decoded(trimmed, torch.stack(attn, dim=1), torch.stack(top_i, dim=1), torch.stack(top_p, dim=1))

    @classmethod
    def from_config(cls, config: Dict) -> "Seq2SeqAttention":
        return cls(**config)
