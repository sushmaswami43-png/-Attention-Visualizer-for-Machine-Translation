"""Produce report figures and per-head attention statistics from the trained checkpoint."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app import load_checkpoint
from data import read_parallel, tokenize

OUT = Path("outputs")
model, src_vocab, tgt_vocab = load_checkpoint(OUT / "model.pt")


def translate(sentence_tokens):
    ids = src_vocab.encode(sentence_tokens) + [src_vocab.eos]
    dec = model.translate(torch.tensor([ids]), torch.tensor([len(ids)]))
    tgt = tgt_vocab.decode(dec.ids[0])
    return sentence_tokens + ["<eos>"], tgt, dec.attention[0, :len(tgt)].numpy()  # [T, h, S]


def heatmap(ax, mat, src, tgt, title):
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(src))); ax.set_xticklabels(src, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt))); ax.set_yticklabels(tgt, fontsize=8)
    ax.set_title(title, fontsize=9)


# 1) Figure: average + each head for one sentence (mirrors the web UI)
sentence = "Ein kleines Mädchen fährt mit dem Fahrrad die Straße entlang."
src, tgt, attn = translate(tokenize(sentence))
n_heads = attn.shape[1]
fig, axes = plt.subplots(1, n_heads + 1, figsize=(3.3 * (n_heads + 1), 4))
heatmap(axes[0], attn.mean(1), src, tgt, "Average of heads")
for h in range(n_heads):
    heatmap(axes[h + 1], attn[:, h], src, tgt, f"Head {h + 1}")
plt.tight_layout(); plt.savefig(OUT / "fig_heads.png", dpi=130); plt.close()

# 2) Per-head statistics over 500 validation sentences
d = Path("data")
pairs = read_parallel(d / "val.de", d / "val.en", 30)[:500]
peaks, entropies, eos_mass, diag = [[] for _ in range(n_heads)], [[] for _ in range(n_heads)], [[] for _ in range(n_heads)], []
for s_tok, _ in pairs:
    s, t, a = translate(s_tok)
    for h in range(n_heads):
        w = a[:, h]
        peaks[h].append(w.max(1).mean())
        entropies[h].append((-(w * np.log(w + 1e-9)).sum(1)).mean())
        eos_mass[h].append(w[:, -1].mean())
    # monotonicity of average-attention argmax (how often the focus moves forward or stays)
    am = a.mean(1).argmax(1)
    diag.append(np.mean(np.diff(am) >= 0) if len(am) > 1 else 1.0)
stats = {
    "per_head": [{"head": h + 1, "mean_peak_weight": float(np.mean(peaks[h])), "mean_entropy_nats": float(np.mean(entropies[h])),
                  "mean_weight_on_eos": float(np.mean(eos_mass[h]))} for h in range(n_heads)],
    "monotonic_focus_fraction": float(np.mean(diag)), "n_sentences": len(pairs)}
(OUT / "attention_stats.json").write_text(json.dumps(stats, indent=2))

# 3) Qualitative samples (val) with translations and reference
samples = []
for (s_tok, ref) in pairs[:8]:
    _, t, _ = translate(s_tok)
    samples.append({"source": " ".join(s_tok), "model": " ".join(x for x in t if x != "<eos>"), "reference": " ".join(ref)})
(OUT / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))

# 4) Loss curves
hist = json.loads((OUT / "metrics.json").read_text())["history"]
plt.figure(figsize=(5, 3.2))
plt.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], marker="o", label="train")
plt.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], marker="o", label="validation")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy (label smoothing 0.1)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig_loss.png", dpi=150)
print(json.dumps(stats, indent=2))
