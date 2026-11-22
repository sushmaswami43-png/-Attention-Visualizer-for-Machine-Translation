import json, textwrap
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = "/mnt/user-data/outputs/Attention_Visualizer_NMT_Assignment.pdf"
O = Path("outputs")
metrics = json.loads((O / "metrics.json").read_text())
stats = json.loads((O / "attention_stats.json").read_text())
samples = json.loads((O / "samples.json").read_text())
hist = metrics["history"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=12, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=27, textColor=colors.HexColor("#1F3A5F"))
SUB = ParagraphStyle("s", parent=BODY, alignment=1, textColor=colors.grey, fontSize=11)
BUL = ParagraphStyle("b", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.8, leading=11, spaceAfter=0)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.2, backColor=colors.HexColor("#F4F6F8"), borderPadding=4, leftIndent=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.5, textColor=colors.grey, alignment=1)

P = lambda t: Paragraph(t, BODY)
B = lambda t: Paragraph(t, BUL, bulletText="\u2022")
M = lambda t: f"<font name='Courier'>{t}</font>"

def table(rows, widths):
    t = Table([[Paragraph(str(c), CELL) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B8C1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t

def listing(title, path):
    lines = []
    for line in Path(path).read_text().rstrip().splitlines():
        indent = len(line) - len(line.lstrip())
        lines += textwrap.wrap(line, 112, subsequent_indent=" " * (indent + 4), replace_whitespace=False, drop_whitespace=False) or [""]
    return [Paragraph(title, H2), Preformatted("\n".join(lines), CODE)]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(20 * mm, 10 * mm, "Attention Visualizer for Machine Translation")
    c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()

ph = stats["per_head"]
cmp_ = json.loads((O / "compare.json").read_text())
A, Bs = cmp_["bleu"]["With attention"], cmp_["bleu"]["No attention (baseline)"]
bins = list(cmp_["sentences_per_bin"])
keep = lambda r: r["17-30"] / r["9-12"] * 100      # longest bin as % of the mid-length bin
NEW = [
    Paragraph("6. Does Attention Help? Baseline Comparison", H1),
    P("To measure what attention buys, the same code was trained with attention switched off "
      f"({M('--no-attention')}). The baseline keeps the encoder, decoder GRU, embeddings, optimiser, seed and 7-epoch budget, "
      "but its decoder only sees the encoder's final state, as in the classic seq2seq model. The attention-specific layers add about "
      "0.3M of the 6.0M parameters. Both runs were scored on the same 998 test sentences, split by source length."),
    table([["Source length (words)", "Sentences", "BLEU with attention", "BLEU, no attention"]] +
          [[b, cmp_["sentences_per_bin"][b], f"{A[b]:.1f}", f"{Bs[b]:.1f}"] for b in bins] +
          [["<b>All</b>", 998, f"<b>{A['overall']:.1f}</b>", f"<b>{Bs['overall']:.1f}</b>"]], [48 * mm, 28 * mm, 47 * mm, 47 * mm]),
    Spacer(1, 4),
    Image(str(O / "fig_compare.png"), width=110 * mm, height=58 * mm), Paragraph("Figure 3. Test BLEU by source length.", CAP),
    P(f"<b>Reading the result.</b> Attention wins in every length bin. The gap also widens with length: on the longest "
      f"sentences (17-30 words) the attention model keeps {keep(A):.0f}% of its 9-12 word score, while the baseline keeps only "
      f"{keep(Bs):.0f}%. This matches the usual explanation that a single fixed-size summary vector becomes a bottleneck for long inputs."),
    P(f"<b>Caveats.</b> The overall gap ({A['overall']:.1f} vs {Bs['overall']:.1f} BLEU) is very large, partly because the baseline is "
      f"under-trained: its best validation loss was still falling at epoch 7 ({cmp_['bleu']['No attention (baseline)']['best_val_loss']:.2f} vs "
      f"{A['best_val_loss']:.2f} with attention), so attention here also speeds up learning and a longer run would narrow the gap. "
      "Results are from one seed, the longest bin has only 132 sentences, and BLEU is on tokenised, lower-cased text (see Section 4)."),
    Paragraph("7. Uncertainty: Top-k Candidates", H1),
    P("At every step the decoder records the five most probable next words. The UI colours each translated word by the probability of the "
      "word that was chosen (green 70% or more, amber 40-70%, red below 40%), and pinning a word shows its top-5 alternatives as bars. "
      "Low-confidence words are the natural places to look for errors: unknown source words, ambiguous words, or places where attention is "
      "spread out (check the focus bar in the heatmap). Two caveats: the probabilities come from a softmax trained with label smoothing "
      "(0.1), so they are deliberately under-confident, and no calibration analysis was done, so treat the colours as a relative "
      "signal, not as true probabilities."),
    Paragraph("8. Training Dynamics: How Attention Is Learned", H1),
    Image(str(O / "fig_evolution.png"), width=172 * mm, height=48 * mm),
    Paragraph("Figure 4. Head-averaged attention for one sentence after epochs 1, 2, 3, 5 and 7.", CAP),
    P("At epoch 1 attention is blurry and the translation is poor and repetitive (\"a little girl is sitting with the street with the street\"). "
      "By epoch 2-3 a diagonal alignment is visible and the verb and nouns are translated correctly, and epochs 5 and 7 look "
      "nearly identical, which agrees with the loss curve flattening. The snapshots come from a second training run with identical settings and seed; "
      "its metrics matched the first run exactly (validation BLEU 34.7, test BLEU 33.5), so they correspond to the deployed model. "
      "This is one sentence and descriptive only."),
]
s = [Spacer(1, 28 * mm), Paragraph("Attention Visualizer for Machine Translation", TITLE),
     Paragraph("Bi-GRU encoder, multi-head attention decoder, FastAPI service and interactive heatmap", SUB),
     Spacer(1, 6 * mm), Paragraph("German to English &nbsp;|&nbsp; Multi30k (28,908 training pairs) &nbsp;|&nbsp; PyTorch", SUB), Spacer(1, 10 * mm)]

s += [Paragraph("1. Objective and Deliverables", H1),
      P("Attention lets a decoder look back at the source sentence each time it emits a word, and the weights it "
        "uses are a direct window into model behaviour. This project trains a recurrent translation model with "
        "multi-head attention and wraps it in a web tool where a user types a German sentence and sees an "
        "attention heatmap aligned with the English output."),
      table([["Requirement", "Status"],
             ["Recurrent seq2seq with attention, corpus of 10,000+ pairs", f"Done. Multi30k de-en, {metrics['train_pairs']:,} training pairs after length filtering. Model trained for 7 epochs on CPU."],
             ["Web heatmap for user-entered sentences", "Done. Single-page UI (vanilla JS) served by the API; tested end-to-end against the live API."],
             ["Controls for heads / layers", "Done for heads: Average view plus one button per head (4 heads), each labelled with its peak sharpness. The model has one recurrent layer, so there is no layer selector."],
             ["Lightweight API", "Done. FastAPI, one POST endpoint plus a health check, CPU inference."],
             ["Extensions (added)", "No-attention baseline compared by sentence length (Section 6), top-k candidates with confidence colouring (Section 7), attention-during-training view (Section 8), and an offline demo mode in the web page."],
             ["Public URL", "<b>Not done in this submission.</b> The sandbox could not publish a site. A Dockerfile and step-by-step deployment instructions (Section 7) are included; hosting takes about 10 minutes."]],
            [62 * mm, 108 * mm]),
      Paragraph("2. Data and Preprocessing", H1),
      P("<b>Corpus.</b> Multi30k task 1 (Flickr30k image captions with German and English sentences): 29,000 "
        f"training, 1,014 validation and 1,000 test (2016 Flickr) pairs. After tokenising and dropping empty pairs "
        f"or sentences longer than 30 tokens, the splits contain {metrics['train_pairs']:,} / 1,010 / 998 pairs. "
        f"{M('download_data.py')} fetches it from GitHub."),
      B("<b>Tokenisation:</b> lower-case, then a regex splitting words and punctuation. No subword model, to keep the pipeline transparent."),
      B("<b>Vocabularies:</b> tokens seen at least twice, capped at 8,000 (German, 7,833 incl. specials) and 6,000 (English, 5,871). Everything else maps to &lt;unk&gt;."),
      B("<b>Special tokens:</b> source gets a trailing &lt;eos&gt;; target is &lt;sos&gt; ... &lt;eos&gt;. The source &lt;eos&gt; stays visible in the heatmap because the model often parks attention on it for punctuation and end-of-sentence."),
      B("<b>Batching:</b> a bucket sampler sorts sentences by length inside chunks of 50 batches so padding (and wasted compute) stays small."),
      Paragraph("3. Model Architecture", H1),
      table([["Part", "Details"],
             ["Encoder", "Embedding(192) then bidirectional GRU (hidden 256 per direction) with packed sequences. Outputs are projected to 256 dims (tanh) to form the attention memory. Initial decoder state = tanh(Linear(forward and backward final states))."],
             ["Attention", "Multi-head scaled dot-product attention, 4 heads of 64 dims. Query = decoder state; keys/values = projections of encoder memory (computed once per sentence). Padding is masked with -inf before softmax. Returns the context vector and the per-head weights [heads x source]."],
             ["Decoder", "GRUCell with input feeding: input = [embedding of previous word ; previous context]. Output features = tanh(Linear([state ; context])), then a linear layer to the vocabulary."],
             ["Size / loss", f"6.03M parameters. Cross-entropy with label smoothing 0.1 (padding ignored). Adam, lr 2e-3, decayed by 0.85 per epoch, gradient clipping 1.0, dropout 0.3, batch size 128."],
             ["Decoding", "Greedy decoding up to 50 steps. The attention weights and top-5 candidate probabilities of every step are returned to the visualiser."]],
            [28 * mm, 142 * mm]),
      Spacer(1, 4),
      P("At step t the decoder state is <i>s<sub>t</sub></i>. For head <i>h</i>, the weight on source position "
        "<i>j</i> is softmax<sub>j</sub>( (W<sub>q</sub><sup>h</sup> s<sub>t</sub>) &middot; (W<sub>k</sub><sup>h</sup> m<sub>j</sub>) / sqrt(d<sub>k</sub>) ), "
        "where <i>m<sub>j</sub></i> is the encoder memory at position j. Each head's context is the weighted sum "
        "of its value projections; heads are concatenated and mixed by an output projection. Because every head "
        "has its own softmax, each one can learn a different alignment behaviour, which is exactly what the "
        "head selector in the UI lets you compare."),
      Paragraph("4. Training Results", H1),
      P("Training took about 19 minutes on one CPU core. Validation loss fell every epoch, and the final train/validation "
        "gap is small, so the model is still under-trained rather than over-fitted; more epochs or a larger model would "
        "likely help."),
      table([["Epoch", "Train loss", "Val loss"]] + [[h["epoch"], f"{h['train_loss']:.3f}", f"{h['val_loss']:.3f}"] for h in hist],
            [30 * mm, 40 * mm, 40 * mm]),
      Spacer(1, 6),
      table([["Metric", "Value"],
             ["BLEU, validation (greedy)", f"{metrics['val_bleu']:.1f}"],
             ["BLEU, test 2016 Flickr (greedy)", f"{metrics['test_bleu']:.1f}"]], [70 * mm, 40 * mm]),
      P("<i>BLEU caveat:</i> computed with sacreBLEU on lower-cased, <b>tokenised</b> text (no detokenisation, "
        "tokenize='none'), using greedy search and a 30-token length filter. It is a consistent internal measure, "
        "but is not directly comparable with published detokenised BLEU scores."),
      Image(str(O / "fig_loss.png"), width=88 * mm, height=56 * mm), Paragraph("Figure 1. Training and validation loss.", CAP)]

s += [Paragraph("Sample translations (first validation sentences, not cherry-picked)", H2),
      table([["German source", "Model output", "Reference"]] + [[x["source"], x["model"], x["reference"]] for x in samples[:6]],
            [60 * mm, 55 * mm, 55 * mm]),
      Spacer(1, 4),
      P("Short, common captions are translated well. Failures are typical of a small word-level model: rare "
        "words become &lt;unk&gt; (\"baumwolle\", \"eisfischerh\u00fctte\"), and long sentences drift or hallucinate details "
        "(\"striped shirt\" for a life jacket). These failures also show up in the heatmap, which is the point of the tool.")]

s += [Paragraph("5. Reading the Attention Patterns", H1),
      Image(str(O / "fig_heads.png"), width=172 * mm, height=44 * mm),
      Paragraph("Figure 2. Attention for one sentence: average over heads, then heads 1-4 (rows = English words, columns = German words). "
                "Rendered with matplotlib from the API's output; the web UI shows the same matrices interactively.", CAP),
      P("<b>What the figure shows.</b> The model learns near-diagonal alignments: <i>girl</i> attends to "
        "<i>m\u00e4dchen</i>, <i>rides</i> to <i>f\u00e4hrt</i>, <i>bike</i> to <i>fahrrad</i>, <i>street</i> to <i>stra\u00dfe</i>, and the final "
        "full stop and &lt;eos&gt; look at the German full stop and &lt;eos&gt;. The word <i>down</i> is the interesting "
        "case: English puts it before the noun phrase while German puts <i>entlang</i> after it, and head 2 puts its mass "
        "on <i>entlang</i>, an example of reordering that a purely monotonic alignment would miss."),
      P("<b>The heads differ.</b> Heads 1 and 4 are sharp and almost one-hot. Head 1 is shifted by one position at the "
        "start of the sentence (<i>a</i> and <i>little</i> both point at <i>kleines</i>), whereas head 4 aligns <i>a</i> with <i>ein</i> "
        "correctly. In this example head 3 is diffuse and spreads over the verb and nearby words, acting more like a context gatherer "
        "than an aligner. Statistics over 500 validation sentences back this up:"),
      table([["Head", "Mean peak weight", "Mean entropy (nats)", "Mean weight on &lt;eos&gt;"]] +
            [[h["head"], f"{h['mean_peak_weight']:.2f}", f"{h['mean_entropy_nats']:.2f}", f"{h['mean_weight_on_eos']:.2f}"] for h in ph],
            [25 * mm, 45 * mm, 50 * mm, 50 * mm]),
      Spacer(1, 4),
      P(f"Across the same sentences the focus of the head-averaged attention moves forward or stays in place in "
        f"{stats['monotonic_focus_fraction'] * 100:.0f}% of consecutive steps, so German-English alignment here is mostly monotonic "
        "with occasional local reordering. These interpretations come from one example plus summary statistics; "
        "they are descriptive, not a proof of what each head \"means\"."),
      P("<b>Attention is not a full explanation.</b> High weight shows where the decoder read from, not that the "
        "word caused the output. The visualiser is best used to spot suspicious behaviour, such as attention "
        "scattered across the sentence when the model emits an <i>&lt;unk&gt;</i> or a hallucinated word."),
      *NEW,
      Paragraph("9. Web Application", H1),
      table([["Item", "Description"],
             ["POST /api/translate", "Body: {\"sentence\": \"...\"}. Returns source_tokens, target_tokens, translation, unknown_tokens, n_heads, attention as [target step][head][source position], and alternatives (top-5 words with probabilities per step). Validation: 1-300 characters, at most 40 words; clear 422 errors otherwise."],
             ["GET /api/health", "Liveness check for the hosting platform."],
             ["GET /", "Serves the single-page UI. The model is loaded once at startup."],
             ["Heatmap", "Rows = English words, columns = German words, cell colour = weight. Optional numeric values, and an outline marking the strongest source word per row."],
             ["Head controls", "Average or Head 1-4. Each button shows the head's mean peak weight so sharp and diffuse heads are easy to spot."],
             ["Linked highlighting", "Hovering a cell or an English word shades the source sentence by attention weight and underlines the matching row and column labels."],
             ["Top candidates", "Words in the translation are coloured by model confidence. Pinning a word (click, or arrow keys / the step buttons) lists its top-5 alternatives with probabilities."],
             ["Alignment lines", "Second view: lines between source and target words, thickness by weight, with a minimum-weight slider. The pinned word's lines stay emphasised."],
             ["Training progress", "For the example sentences, a slider and Play button step through the attention learned after epochs 1, 2, 3, 5 and 7."],
             ["Demo mode", "If no API is reachable (for example the page is hosted statically), the page switches to saved model outputs for the example sentences and shows a banner instead of failing. Custom sentences need the Python server."],
             ["Robustness", "Unknown words are dashed in the source sentence with a note. Text is inserted via textContent (no HTML injection). Dark mode and keyboard focus styles are supported."]],
            [38 * mm, 132 * mm]),
      Spacer(1, 4),
      P("<b>Testing performed.</b> The API was run with uvicorn and queried: attention rows sum to 1.0 for every "
        "head, empty input returns 422. The page was then loaded in a headless DOM (jsdom) against the live API: "
        "4 example chips rendered; an example produced an 11 x 10 grid (110 cells); switching heads changed the "
        "matrix and the pressed button; value labels, argmax outlines and hover highlighting worked. "
        "<b>It has not been checked in a real browser</b>, so visual polish (spacing, colours) is unverified."),
      P("<b>Later additions tested the same way:</b> in the jsdom run, confidence colours and the top-5 panel rendered in both live and demo mode, "
        "the training slider appeared for example sentences (5 epochs), hid for a custom sentence, and Play stepped through epochs; no script errors."),
      Paragraph("10. Running and Deploying", H1),
      Preformatted("pip install torch fastapi \"uvicorn[standard]\" sacrebleu matplotlib\n"
                   "python download_data.py             # fetch Multi30k\n"
                   "python train.py --epochs 7           # writes outputs/model.pt and metrics.json (~20 min CPU)\n"
                   "uvicorn app:app --port 8000          # open http://localhost:8000\n"
                   "python analyze.py                    # optional: figures and per-head statistics\n"
                   "# extras: baseline, snapshots, comparison, and the page's demo data\n"
                   "python train.py --no-attention --out-dir outputs/baseline_run\n"
                   "python train.py --out-dir outputs/attn_run --snapshot-epochs 1 2 3 5 7\n"
                   "python compare.py                    # BLEU by sentence length\n"
                   "python embed_demo.py                 # refresh the page's demo data", CODE),
      Spacer(1, 6),
      P("<b>Public hosting (Hugging Face Spaces, free CPU tier).</b> 1) Create a new Space and choose the Docker SDK. "
        "2) Put these files in the repo: data.py, model.py, app.py, static/, Dockerfile, outputs/model.pt. 3) Add this "
        "header to README.md: <font name='Courier'>sdk: docker</font> and <font name='Courier'>app_port: 7860</font>. 4) Because "
        "model.pt (about 24 MB) exceeds the 10 MB plain-git limit, run <font name='Courier'>git lfs track \"*.pt\"</font> before committing. "
        "The Space then builds the Dockerfile and publishes a public URL. Render or Railway work the same way "
        "(start command: uvicorn app:app --host 0.0.0.0 --port $PORT)."),
      Paragraph("11. Limitations and Further Work", H1),
      B("<b>Vocabulary:</b> word-level tokens create many &lt;unk&gt;s; byte-pair encoding would fix this but makes heatmap columns sub-word pieces."),
      B("<b>Decoding:</b> greedy only. Beam search would raise BLEU, but attention must then be tracked per hypothesis."),
      B("<b>Layers:</b> one recurrent layer, so the layer selector requested for deeper models is not applicable; a second decoder layer with its own attention would add one."),
      B("<b>Evaluation depth:</b> one seed per model and an under-trained baseline (Section 6), so the size of the attention gain is an upper estimate; multiple seeds and longer training would tighten it."),
      B("<b>Uncertainty:</b> confidence colours are uncalibrated; a reliability diagram and temperature scaling would make them trustworthy as probabilities."),
      B("<b>Interpretation:</b> alignment quality is judged by eye plus summary statistics; no gold word alignments were used."),
      PageBreak(), Paragraph("Appendix: Source Code", H1)]

for title, path in [("A. download_data.py", "download_data.py"), ("B. data.py", "data.py"), ("C. model.py", "model.py"),
                    ("D. train.py", "train.py"), ("E. app.py", "app.py"), ("F. static/index.html", "static/index.html"),
                    ("G. analyze.py", "analyze.py"), ("H. compare.py", "compare.py"),
                    ("I. embed_demo.py", "embed_demo.py"), ("J. Dockerfile", "Dockerfile")]:
    s += listing(title, path)

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                  title="Attention Visualizer for Machine Translation", author="Assignment").build(s, onFirstPage=footer, onLaterPages=footer)
print("built")
