import json
from pathlib import Path

CONFIG_CELL = '''import os, math, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
for d in sorted(os.listdir("/kaggle/input")):
    print("input dataset:", d)
    for f in os.listdir(f"/kaggle/input/{d}"):
        print("   ", f)
'''

CFG_CELL = '''CFG = dict(
    tokens_path="/kaggle/input/qm-corpus-v0/corpus_v0.tokens.bin",
    meta_path="/kaggle/input/qm-corpus-v0/corpus_v0.tokens.meta.json",
    out_dir="/kaggle/working",
    vocab_size=49152,
    d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, ffn_dim=2048,
    max_seq_len=2048, rope_theta=10000.0, norm_eps=1e-5,
    batch_size=16, grad_accum=2,
    lr_peak=8e-4, warmup=200, min_lr_ratio=0.1,
    grad_clip=1.0, weight_decay=0.1,
    max_hours=8.5, hard_cap_steps=20000,
    log_every=25, ckpt_every=500, eval_every=250,
    seed=42,
)
CFG
'''

MODEL_CELL = '''class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt)

def rope_cache(T, hd, theta, dev, dt):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    t = torch.arange(T, device=dev, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos().to(dt), f.sin().to(dt)

def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)

class Attn(nn.Module):
    def __init__(self, d, nh, nkv, theta):
        super().__init__()
        self.nh, self.nkv, self.hd, self.theta = nh, nkv, d // nh, theta
        self.wq = nn.Linear(d, nh * self.hd, bias=False)
        self.wk = nn.Linear(d, nkv * self.hd, bias=False)
        self.wv = nn.Linear(d, nkv * self.hd, bias=False)
        self.wo = nn.Linear(nh * self.hd, d, bias=False)
    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        r = self.nh // self.nkv
        if r > 1:
            k = k.repeat_interleave(r, 1)
            v = v.repeat_interleave(r, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(b, t, -1))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["d_model"]
        self.n1 = RMSNorm(d, c["norm_eps"])
        self.attn = Attn(d, c["n_heads"], c["n_kv_heads"], c["rope_theta"])
        self.n2 = RMSNorm(d, c["norm_eps"])
        self.gate = nn.Linear(d, c["ffn_dim"], bias=False)
        self.up = nn.Linear(d, c["ffn_dim"], bias=False)
        self.down = nn.Linear(c["ffn_dim"], d, bias=False)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.down(F.silu(self.gate(self.n2(x))) * self.up(self.n2(x)))

class QMModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c["vocab_size"], c["d_model"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layers"]))
        self.norm = RMSNorm(c["d_model"], c["norm_eps"])
        self.head = nn.Linear(c["d_model"], c["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        b, t = idx.shape
        hd = self.c["d_model"] // self.c["n_heads"]
        cos, sin = rope_cache(t, hd, self.c["rope_theta"], idx.device, self.emb.weight.dtype)
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss
'''

DATA_CELL = '''torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.memmap(CFG["tokens_path"], dtype=np.uint16, mode="r")
rng = np.random.default_rng(CFG["seed"])
L = CFG["max_seq_len"]
print("tokens available:", len(data))

def get_batch(bs):
    hi = len(data) - L - 1
    off = rng.integers(0, hi + 1, size=bs)
    x = np.stack([data[o:o + L] for o in off])
    y = np.stack([data[o + 1:o + 1 + L] for o in off])
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device))

model_cfg = {k: CFG[k] for k in ["vocab_size","d_model","n_layers","n_heads","n_kv_heads","ffn_dim","rope_theta","norm_eps"]}
model = QMModel(model_cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params/1e6:.1f}M on {device}")

opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr_peak"], betas=(0.9, 0.95), weight_decay=CFG["weight_decay"])
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

def lr_at(step):
    if step < CFG["warmup"]:
        return CFG["lr_peak"] * (step + 1) / CFG["warmup"]
    prog = min(1.0, (step - CFG["warmup"]) / max(1, CFG["hard_cap_steps"] - CFG["warmup"]))
    return CFG["lr_peak"] * (CFG["min_lr_ratio"] + (1 - CFG["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog)))

@torch.no_grad()
def eval_loss(n_batches=8):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(CFG["batch_size"])
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

print("initial eval loss:", round(eval_loss(), 4))
'''

TRAIN_CELL = '''loss_history = []
eval_history = []
step = 0
t0 = time.time()
deadline = t0 + CFG["max_hours"] * 3600
model.train()
run_loss = 0.0

while step < CFG["hard_cap_steps"] and time.time() < deadline:
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    for _ in range(CFG["grad_accum"]):
        x, y = get_batch(CFG["batch_size"])
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, targets=y)
        scaler.scale(loss / CFG["grad_accum"]).backward()
        run_loss += loss.item() / CFG["grad_accum"]
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
    scaler.step(opt)
    scaler.update()

    if not math.isfinite(run_loss):
        raise RuntimeError(f"non-finite loss at step {step}")

    if step % CFG["log_every"] == 0:
        elapsed = time.time() - t0
        toks = step * CFG["batch_size"] * CFG["grad_accum"] * L
        print(f"step {step:6d} | loss {run_loss:.4f} | lr {lr:.2e} | {toks/1e6:.0f}M tok | {elapsed/60:.1f} min")
        loss_history.append({"step": step, "loss": run_loss})
    run_loss = 0.0

    if (step + 1) % CFG["eval_every"] == 0:
        ev = eval_loss()
        eval_history.append({"step": step, "eval_loss": ev})
        print(f"  [eval] step {step}: loss {ev:.4f}")

    if (step + 1) % CFG["ckpt_every"] == 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": CFG},
                   f"{CFG['out_dir']}/qm_pilot_ckpt.pt")

    step += 1

torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": CFG},
           f"{CFG['out_dir']}/qm_pilot_final.pt")
final_eval = eval_loss()
print(f"TRAINING DONE | steps {step} | final eval loss {final_eval:.4f} | {(time.time()-t0)/60:.1f} min")
'''

REPORT_CELL = '''import matplotlib.pyplot as plt

report = {
    "cfg": {k: v for k, v in CFG.items()},
    "params_millions": round(n_params / 1e6, 1),
    "steps_completed": step,
    "tokens_seen_millions": round(step * CFG["batch_size"] * CFG["grad_accum"] * L / 1e6, 1),
    "final_eval_loss": final_eval,
    "loss_history": loss_history,
    "eval_history": eval_history,
}
with open(f"{CFG['out_dir']}/pilot_report.json", "w") as f:
    json.dump(report, f, indent=2)

if loss_history:
    xs = [h["step"] for h in loss_history]
    ys = [h["loss"] for h in loss_history]
    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys, label="train loss")
    if eval_history:
        plt.plot([h["step"] for h in eval_history], [h["eval_loss"] for h in eval_history], "o-", label="eval loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("Quantsmind pilot-113M")
    plt.savefig(f"{CFG['out_dir']}/loss_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
print("outputs:", os.listdir(CFG["out_dir"]))
'''

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = [
    md_cell("# Quantsmind — 300M-class Pilot Pretrain (Kaggle free GPU)\n\n"
            "Trains **your** from-scratch model (RMSNorm + RoPE + GQA + SwiGLU) on **your** corpus tokens.\n\n"
            "**Before Run All:**\n"
            "1. Upload `corpus_v0.tokens.bin` + `corpus_v0.tokens.meta.json` as a Kaggle Dataset named `qm-corpus-v0`\n"
            "2. Settings → Accelerator → **GPU T4 x2** (or P100) · Internet **OFF** is fine\n"
            "3. Edit `tokens_path`/`meta_path` in the config cell to match your dataset slug\n\n"
            "The run self-stops at the time budget and saves everything to `/kaggle/working`."),
    md_cell("## 1. Environment check"),
    code_cell(CONFIG_CELL),
    md_cell("## 2. Config — edit the two input paths to your dataset slug"),
    code_cell(CFG_CELL),
    md_cell("## 3. Model (your architecture: RMSNorm · RoPE · GQA · SwiGLU)"),
    code_cell(MODEL_CELL),
    md_cell("## 4. Data + optimizer + eval"),
    code_cell(DATA_CELL),
    md_cell("## 5. Train"),
    code_cell(TRAIN_CELL),
    md_cell("## 6. Report + loss curve"),
    code_cell(REPORT_CELL),
    md_cell("## Next\n\n"
            "Download from `/kaggle/working`: `qm_pilot_final.pt`, `pilot_report.json`, `loss_curve.png`.\n\n"
            "Success signal for this pilot: eval loss trending below ~4.5 and still falling. "
            "Attach `pilot_report.json` to the repo `experiments/` notes when done."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_pilot.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
json.loads(out.read_text(encoding="utf-8"))
print(f"wrote {out} ({out.stat().st_size} bytes, valid JSON)")
