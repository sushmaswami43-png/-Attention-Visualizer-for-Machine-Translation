"""Train the attention NMT model on Multi30k (de->en, 29k pairs) and report BLEU.

Usage: python train.py --data-dir data --epochs 8
"""
import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BucketBatchSampler, TranslationDataset, Vocab, read_parallel
from model import Seq2SeqAttention


def make_loader(ds: TranslationDataset, batch_size: int, shuffle: bool) -> DataLoader:
    sampler = BucketBatchSampler([len(s) for s in ds.src], batch_size, shuffle)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=ds.collate)


def run_epoch(model, loader, criterion, device, optimizer=None, clip: float = 1.0) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for src, src_len, tgt in loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            logits = model(src, src_len, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            n_tok = (tgt[:, 1:] != model.config["tgt_pad"]).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
    return total_loss / total_tokens


def compute_bleu(model, pairs, src_vocab, tgt_vocab, device, batch_size: int = 64) -> float:
    """Corpus BLEU on lower-cased, tokenised text (so it is not comparable to detokenised BLEU)."""
    hyps, refs = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        encoded = [src_vocab.encode(s) + [src_vocab.eos] for s, _ in chunk]
        lens = torch.tensor([len(e) for e in encoded])
        src = torch.full((len(chunk), int(lens.max())), src_vocab.pad, dtype=torch.long)
        for j, e in enumerate(encoded):
            src[j, :len(e)] = torch.tensor(e)
        ids = model.translate(src.to(device), lens.to(device)).ids
        for row, (_, ref) in zip(ids, chunk):
            hyps.append(" ".join(tgt_vocab.decode([t for t in row if t != tgt_vocab.eos])))
            refs.append(" ".join(ref))
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-attention", action="store_true", help="train the classic seq2seq baseline")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[], help="also save weights after these epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = args.data_dir
    train_pairs = read_parallel(d / "train.de", d / "train.en", args.max_len)
    val_pairs = read_parallel(d / "val.de", d / "val.en", args.max_len)
    test_pairs = read_parallel(d / "test_2016_flickr.de", d / "test_2016_flickr.en", args.max_len)
    print(f"pairs: train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

    src_vocab = Vocab.build([s for s, _ in train_pairs], min_freq=2, max_size=8000)
    tgt_vocab = Vocab.build([t for _, t in train_pairs], min_freq=2, max_size=6000)
    train_dl = make_loader(TranslationDataset(train_pairs, src_vocab, tgt_vocab), args.batch_size, True)
    val_dl = make_loader(TranslationDataset(val_pairs, src_vocab, tgt_vocab), args.batch_size, False)

    model = Seq2SeqAttention(len(src_vocab), len(tgt_vocab), src_vocab.pad, tgt_vocab.pad,
                             tgt_vocab.sos, tgt_vocab.eos, n_heads=args.heads,
                             use_attention=not args.no_attention).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad, label_smoothing=0.1)

    best, history, first_epoch = float("inf"), [], 1
    resume_path = args.out_dir / "resume.pt"
    if resume_path.exists():  # continue after an interrupted run (weights, optimiser, schedule and RNG state)
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); torch.set_rng_state(state["torch_rng"])
        train_dl.batch_sampler.rng.setstate(state["sampler_rng"])
        best, history, first_epoch = state["best"], state["history"], state["epoch"] + 1
        print(f"resumed after epoch {state['epoch']}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        start = time.time()
        tr = run_epoch(model, train_dl, criterion, device, optimizer)
        va = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch:02d} | train {tr:.3f} | val {va:.3f} | {time.time() - start:.0f}s", flush=True)
        if epoch in args.snapshot_epochs:
            torch.save(model.state_dict(), args.out_dir / f"snapshot_epoch{epoch}.pt")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "config": model.config,
                        "src_vocab": src_vocab.itos, "tgt_vocab": tgt_vocab.itos}, args.out_dir / "model.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "sampler_rng": train_dl.batch_sampler.rng.getstate(),
                    "best": best, "history": history, "epoch": epoch}, resume_path)

    ckpt = torch.load(args.out_dir / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = {"attention": not args.no_attention, "train_pairs": len(train_pairs), "val_bleu": compute_bleu(model, val_pairs, src_vocab, tgt_vocab, device),
               "test_bleu": compute_bleu(model, test_pairs, src_vocab, tgt_vocab, device), "history": history}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
