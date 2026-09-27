import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.local_inference import load_model

app = FastAPI(title="Quantsmind Coder — Completion API")

_model = None
_tokenizer = None
_device = None


class CompletionRequest(BaseModel):
    prefix: str
    suffix: str = ""
    language: str = "python"
    max_tokens: int = 64
    temperature: float = 0.2
    top_k: int = 5


class CompletionResponse(BaseModel):
    completions: list[str]
    scores: list[float]
    model: str


def get_model():
    global _model, _tokenizer, _device
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load first.")
    return _model, _tokenizer, _device


@app.post("/load")
async def load(ckpt: str, tokenizer: str = "tokenizer/qm_code_bpe_v0.json"):
    global _model, _tokenizer, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _ = load_model(ckpt, _device)
    from tokenizers import Tokenizer
    _tokenizer = Tokenizer.from_file(tokenizer)
    return {"status": "loaded", "device": _device}


@app.post("/complete", response_model=CompletionResponse)
async def complete(request: CompletionRequest):
    model, tokenizer, device = get_model()

    if request.suffix:
        prompt = f"<|fim_prefix|>{request.prefix}<|fim_suffix|>{request.suffix}<|fim_middle|>"
    else:
        prompt = request.prefix

    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    completions = []
    scores = []

    with torch.no_grad():
        for _ in range(request.top_k):
            gen_ids = x.clone()
            log_prob_sum = 0.0
            for _ in range(request.max_tokens):
                idx_cond = gen_ids if gen_ids.size(1) <= model.cfg["max_seq_len"] else gen_ids[:, -model.cfg["max_seq_len"]:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / request.temperature
                v, _ = torch.topk(logits, min(50, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                log_prob_sum += torch.log(probs[0, next_id.item()] + 1e-10).item()
                gen_ids = torch.cat([gen_ids, next_id], dim=1)
                if next_id.item() == tokenizer.token_to_id("<eos>"):
                    break

            generated = tokenizer.decode(gen_ids[0].tolist())
            completion = generated[len(prompt):] if generated.startswith(prompt) else generated
            completions.append(completion)
            scores.append(log_prob_sum / max(1, request.max_tokens))

    return CompletionResponse(
        completions=completions,
        scores=scores,
        model="quantsmind-coder",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
