"""Tokenisation, vocabularies and batching for a parallel corpus (Multi30k de->en)."""
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lower-case and split into word / punctuation tokens."""
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Bidirectional token <-> id mapping. Special tokens always occupy ids 0-3."""

    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, sentences: Sequence[Sequence[str]], min_freq: int = 2,
              max_size: int = 8000) -> "Vocab":
        counts = Counter(tok for sent in sentences for tok in sent)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls(SPECIALS + words)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    @property
    def pad(self) -> int: return self.stoi[PAD]
    @property
    def sos(self) -> int: return self.stoi[SOS]
    @property
    def eos(self) -> int: return self.stoi[EOS]
    def __len__(self) -> int: return len(self.itos)


def read_parallel(src_path: Path, tgt_path: Path, max_len: int = 30) -> List[Tuple[List[str], List[str]]]:
    """Read aligned files, tokenise, and drop empty or over-long pairs."""
    pairs = []
    with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
        for src_line, tgt_line in zip(fs, ft):
            src, tgt = tokenize(src_line), tokenize(tgt_line)
            if 0 < len(src) <= max_len and 0 < len(tgt) <= max_len:
                pairs.append((src, tgt))
    return pairs


class TranslationDataset(Dataset):
    """Holds id-encoded pairs. Source gets <eos>; target gets <sos> ... <eos>."""

    def __init__(self, pairs, src_vocab: Vocab, tgt_vocab: Vocab):
        self.src = [src_vocab.encode(s) + [src_vocab.eos] for s, _ in pairs]
        self.tgt = [[tgt_vocab.sos] + tgt_vocab.encode(t) + [tgt_vocab.eos] for _, t in pairs]
        self.src_pad, self.tgt_pad = src_vocab.pad, tgt_vocab.pad

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int):
        return self.src[i], self.tgt[i]

    def collate(self, batch):
        srcs, tgts = zip(*batch)
        src_len = torch.tensor([len(s) for s in srcs])
        src = torch.full((len(srcs), int(src_len.max())), self.src_pad, dtype=torch.long)
        tgt = torch.full((len(tgts), max(len(t) for t in tgts)), self.tgt_pad, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, :len(s)] = torch.tensor(s)
            tgt[i, :len(t)] = torch.tensor(t)
        return src, src_len, tgt


class BucketBatchSampler:
    """Groups similar-length sentences so padding (and compute) is minimal."""

    def __init__(self, lengths: Sequence[int], batch_size: int, shuffle: bool, seed: int = 0):
        self.lengths, self.batch_size, self.shuffle = lengths, batch_size, shuffle
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[List[int]]:
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)
        chunk = self.batch_size * 50
        batches = []
        for start in range(0, len(idx), chunk):
            block = sorted(idx[start:start + chunk], key=lambda i: self.lengths[i])
            batches += [block[j:j + self.batch_size] for j in range(0, len(block), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size)
