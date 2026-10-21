"""Control: compare the trained checkpoint against random initialisation.

If training did real work, the trained model must score better than a model that
never trained, on the same held-out validation windows. This is the cheapest
honest evidence that weights moved in a useful direction rather than merely
existing on disk.
"""
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data import TokenDataset
from training.model import QMModel
from training.validation import evaluate

SEED = 42
model_cfg = yaml.safe_load(
    open("configs/model/code_m3_3m.yaml", encoding="utf-8")
)["model"]

torch.manual_seed(SEED)
val = TokenDataset("data/m3_code_verified.val.tokens.bin", seed=1234)

torch.manual_seed(SEED)
random_model = QMModel(model_cfg)
random_result = evaluate(random_model, val, 512, batches=8, seed=1234,
                         vocab_size=model_cfg["vocab_size"])

ckpt = torch.load("runs/m3_base/checkpoint_best.pt", map_location="cpu",
                  weights_only=False)
torch.manual_seed(SEED)
trained_model = QMModel(model_cfg)
trained_model.load_state_dict(ckpt["model"])
trained_result = evaluate(trained_model, val, 512, batches=8, seed=1234,
                          vocab_size=model_cfg["vocab_size"])

uniform = torch.log(torch.tensor(float(model_cfg["vocab_size"]))).item()

report = {
    "uniform_random_baseline_loss": uniform,
    "random_init_loss": random_result.loss,
    "random_init_perplexity": random_result.perplexity,
    "trained_loss": trained_result.loss,
    "trained_perplexity": trained_result.perplexity,
    "checkpoint": "runs/m3_base/checkpoint_best.pt",
    "checkpoint_step": ckpt.get("step"),
    "windows": random_result.windows,
    "n_windows": len(random_result.windows),
    "seq_len": 512,
    "batches": 8,
}
report["improvement_nats"] = random_result.loss - trained_result.loss
with open("reports/m3_control_vs_random.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print(f"uniform random baseline : {uniform:.4f}")
print(f"random init loss        : {random_result.loss:.4f}  "
      f"ppl={random_result.perplexity}")
print(f"trained (best ckpt)     : {trained_result.loss:.4f}  "
      f"ppl={trained_result.perplexity}")
print(f"improvement             : {report['improvement_nats']:.4f} nats")
print(f"windows evaluated       : {len(random_result.windows)} (identical for both)")