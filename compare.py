"""Compare the attention model with the no-attention baseline: overall and by source-sentence length (test set)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from app import load_checkpoint
from data import read_parallel
from train import compute_bleu

OUT = Path("outputs")
BINS = [(1, 8), (9, 12), (13, 16), (17, 30)]            # source length in words (excluding <eos>)
RUNS = {"With attention": OUT / "attn_run", "No attention (baseline)": OUT / "baseline_run"}


def plot(counts: dict, results: dict, total: int) -> None:
    """Grouped bar chart of BLEU per source-length bin (re-plottable from compare.json)."""
    labels = list(counts) + ["overall"]
    width = 0.38
    plt.figure(figsize=(6.4, 3.4))
    for k, (name, row) in enumerate(results.items()):
        plt.bar([i + (k - 0.5) * width for i in range(len(labels))], [row[l] for l in labels], width, label=name)
    plt.xticks(range(len(labels)), [f"{l}\n(n={counts.get(l, total)})" for l in labels], fontsize=8)
    plt.xlabel("Source length in words"); plt.ylabel("BLEU"); plt.ylim(0, 45); plt.legend(fontsize=8, loc='upper right', ncol=2); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_compare.png", dpi=150)



def main() -> None:
    pairs = read_parallel(Path("data/test_2016_flickr.de"), Path("data/test_2016_flickr.en"), 30)
    counts = {f"{lo}-{hi}": sum(lo <= len(p[0]) <= hi for p in pairs) for lo, hi in BINS}
    results = {}
    for name, run_dir in RUNS.items():
        model, src_vocab, tgt_vocab = load_checkpoint(run_dir / "model.pt")
        bleu = lambda subset: compute_bleu(model, subset, src_vocab, tgt_vocab, torch.device("cpu"))
        row = {"overall": bleu(pairs)}
        row.update({f"{lo}-{hi}": bleu([p for p in pairs if lo <= len(p[0]) <= hi]) for lo, hi in BINS})
        row["best_val_loss"] = min(h["val_loss"] for h in json.loads((run_dir / "metrics.json").read_text())["history"])
        results[name] = row
        print(name, {k: round(v, 2) for k, v in row.items()})
    (OUT / "compare.json").write_text(json.dumps({"sentences_per_bin": counts, "bleu": results}, indent=2))

    plot(counts, results, len(pairs))

if __name__ == "__main__":
    main()
