"""Embed saved model outputs (translations, attention, top-k candidates, training snapshots) into static/index.html.

This powers the page's offline demo mode. Re-run after retraining.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from app import load_checkpoint
from data import EOS, UNK, tokenize
from model import Seq2SeqAttention

OUT, HTML = Path("outputs"), Path("static/index.html")
SNAPSHOT_EPOCHS = [1, 2, 3, 5, 7]
EXAMPLES = ["Ein Mann sitzt auf einer Bank im Park.", "Zwei Hunde spielen im Schnee.",
            "Eine Frau in einem roten Kleid steht vor einem Gebäude.",
            "Ein kleines Mädchen fährt mit dem Fahrrad die Straße entlang.",
            "Zwei Männer arbeiten auf einer Baustelle.", "Ein Hund läuft durch das Gras."]


def r2(x: float):
    """Round to 2 decimals and drop '.0' so the embedded JSON stays small."""
    v = round(float(x), 2)
    return int(v) if v == int(v) else v


def decode(model, src_vocab, tgt_vocab, tokens):
    ids = src_vocab.encode(tokens) + [src_vocab.eos]
    dec = model.translate(torch.tensor([ids]), torch.tensor([len(ids)]))
    return ids, tgt_vocab.decode(dec.ids[0]), dec


def main() -> None:
    model, src_vocab, tgt_vocab = load_checkpoint(OUT / "model.pt")
    snapshot_dir = OUT / "attn_run"
    snap_models = {}
    for e in SNAPSHOT_EPOCHS:
        path = snapshot_dir / f"snapshot_epoch{e}.pt"
        if path.exists():
            m = Seq2SeqAttention.from_config(model.config)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            snap_models[e] = m.eval()

    demo = {}
    for text in EXAMPLES:
        tokens = tokenize(text)
        ids, target, dec = decode(model, src_vocab, tgt_vocab, tokens)
        steps = len(target)
        entry = {
            "text": text, "source_tokens": tokens + [EOS], "target_tokens": target,
            "translation": " ".join(t for t in target if t != EOS),
            "unknown_tokens": [t for t, i in zip(tokens, ids) if src_vocab.itos[i] == UNK],
            "n_heads": dec.attention.size(2),
            "attention": [[[r2(x) for x in head] for head in step] for step in dec.attention[0, :steps].tolist()],
            "alternatives": [[{"token": tgt_vocab.itos[i], "prob": r2(p)} for i, p in zip(ik, pk)]
                             for ik, pk in zip(dec.topk_ids[0, :steps].tolist(), dec.topk_probs[0, :steps].tolist())]}
        snaps = []
        for e, m in snap_models.items():
            _, t_e, d_e = decode(m, src_vocab, tgt_vocab, tokens)
            avg = d_e.attention[0, :len(t_e)].mean(1)
            snaps.append({"e": e, "t": t_e, "a": [[r2(x) for x in row] for row in avg.tolist()]})
        if snaps:
            entry["snaps"] = snaps
        demo[" ".join(tokens)] = entry

    block = "// <demo-data>\nconst DEMO = " + json.dumps(demo, ensure_ascii=False, separators=(",", ":")) + ";\n// </demo-data>"
    HTML.write_text(re.sub(r"// <demo-data>.*?// </demo-data>", lambda _: block, HTML.read_text(), flags=re.S))
    print(f"embedded {len(demo)} sentences, {len(snap_models)} snapshots, {len(block) / 1024:.1f} KB")

    # static figure for the report: attention at each saved epoch for one sentence
    first = next((v for v in demo.values() if "snaps" in v and v["text"].endswith("entlang.")), None)
    if first:
        snaps = first["snaps"]
        fig, axes = plt.subplots(1, len(snaps), figsize=(3.2 * len(snaps), 3.6))
        for ax, sn in zip(axes, snaps):
            ax.imshow(sn["a"], cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(first["source_tokens"]))); ax.set_xticklabels(first["source_tokens"], rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(sn["t"]))); ax.set_yticklabels(sn["t"], fontsize=7)
            ax.set_title(f"Epoch {sn['e']}", fontsize=9)
        plt.tight_layout(); plt.savefig(OUT / "fig_evolution.png", dpi=130)


if __name__ == "__main__":
    main()
