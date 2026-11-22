"""FastAPI server: serves the heatmap UI and a JSON translation endpoint.

Run locally:  uvicorn app:app --port 8000
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data import EOS, UNK, Vocab, tokenize
from model import Seq2SeqAttention

MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/model.pt"))
STATIC_DIR = Path(__file__).parent / "static"
MAX_SOURCE_TOKENS = 40
torch.set_num_threads(1)


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model = Seq2SeqAttention.from_config(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), Vocab(ckpt["src_vocab"]), Vocab(ckpt["tgt_vocab"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.src_vocab, app.state.tgt_vocab = load_checkpoint(MODEL_PATH)
    yield


app = FastAPI(title="NMT Attention Visualizer", lifespan=lifespan)
# Lets the UI be hosted on a different origin than the API (set CORS_ORIGINS="https://a.com,https://b.com" to restrict).
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class TranslateRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=300)


class Alternative(BaseModel):
    token: str
    prob: float


class TranslateResponse(BaseModel):
    source_tokens: List[str]
    target_tokens: List[str]
    translation: str
    unknown_tokens: List[str]
    n_heads: int
    attention: List[List[List[float]]]   # [target_step][head][source_position]
    alternatives: List[List[Alternative]]   # [target_step][rank], best first


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest, request: Request):
    model, src_vocab, tgt_vocab = request.app.state.model, request.app.state.src_vocab, request.app.state.tgt_vocab
    tokens = tokenize(req.sentence)
    if not tokens:
        raise HTTPException(status_code=422, detail="Sentence contains no translatable tokens.")
    if len(tokens) > MAX_SOURCE_TOKENS:
        raise HTTPException(status_code=422, detail=f"Please keep the sentence under {MAX_SOURCE_TOKENS} words.")

    ids = src_vocab.encode(tokens) + [src_vocab.eos]
    src = torch.tensor([ids])
    dec = model.translate(src, torch.tensor([len(ids)]))
    target = tgt_vocab.decode(dec.ids[0])
    steps = len(target)
    return TranslateResponse(
        source_tokens=tokens + [EOS],
        target_tokens=target,
        translation=" ".join(t for t in target if t != EOS),
        unknown_tokens=[tok for tok, i in zip(tokens, ids) if src_vocab.itos[i] == UNK],
        n_heads=dec.attention.size(2),
        attention=dec.attention[0, :steps].round(decimals=4).tolist(),
        alternatives=[[Alternative(token=tgt_vocab.itos[i], prob=round(p, 4)) for i, p in zip(ids_k, probs_k)]
                      for ids_k, probs_k in zip(dec.topk_ids[0, :steps].tolist(), dec.topk_probs[0, :steps].tolist())],
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
