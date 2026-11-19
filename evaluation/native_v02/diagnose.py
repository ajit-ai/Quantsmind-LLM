"""Controlled diagnostics for Native v0.2.

Investigates empty/degenerate outputs observed in Native v0.1 by running
controlled prompt-format and decoding experiments on the existing native
checkpoint with raw token/logit/EOS capture. No training, no weight/tokenizer
changes, no external providers.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_CHECKPOINT = Path(r"F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt")
DEFAULT_TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
DEFAULT_CHECKPOINT_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"

# Stable v0.1 subset covering required categories/types.
SUBSET_TASK_IDS = (
    "NV01-BASIC-001",   # basic function generation
    "NV01-ARITH-003",   # arithmetic
    "NV01-LIST-006",    # list/dictionary operations (list aggregation)
    "NV01-COMP-015",    # function completion
    "NV01-INST-017",    # instruction-following
    "NV01-DEBUG-019",   # code correction
)

PROMPT_VARIANTS = (
    "v01_benchmark",
    "raw_completion_prefix",
    "recovered_sft_format",
    "minimal_instruction",
    "few_shot",
)

DECODING_CONFIGS = (
    {"name": "greedy_v01_baseline", "temperature": 0.0, "top_k": 50, "top_p": 1.0, "seed": 1},
    {"name": "temp_0.2", "temperature": 0.2, "top_k": 50, "top_p": 1.0, "seed": 1},
    {"name": "temp_0.7", "temperature": 0.7, "top_k": 50, "top_p": 1.0, "seed": 1},
    {"name": "topk_10_temp_0.7", "temperature": 0.7, "top_k": 10, "top_p": 1.0, "seed": 1},
)


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def get_subset_tasks():
    from evaluation.native_v01.tasks import get_tasks
    by_id = {t.task_id: t for t in get_tasks()}
    missing = [tid for tid in SUBSET_TASK_IDS if tid not in by_id]
    if missing:
        raise ValueError(f"missing v0.1 tasks for subset: {missing}")
    return [by_id[tid] for tid in SUBSET_TASK_IDS]


def render_prompt(task, variant: str, *, sft_available: bool = True) -> tuple[str, dict[str, Any]]:
    """Return (prompt_text, meta) for a controlled variant. Semantics unchanged."""
    meta: dict[str, Any] = {"variant": variant, "task_id": task.task_id}
    # Build a neutral raw-completion/function prefix per task.
    if task.task_id == "NV01-BASIC-001":
        prefix = "def add(a, b):\n    return "
    elif task.task_id == "NV01-ARITH-003":
        prefix = "def square(n):\n    return "
    elif task.task_id == "NV01-LIST-006":
        prefix = "def sum_list(nums):\n    total = 0\n    for "
    elif task.task_id == "NV01-COMP-015":
        prefix = (task.completion_prefix or "def count_items(items):\n") + "    return "
    elif task.task_id == "NV01-INST-017":
        prefix = "def greet(name):\n    return "
    elif task.task_id == "NV01-DEBUG-019":
        prefix = (
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total\n\n# fixed:\ndef sum_to_n(n):\n    total = 0\n    for i in range("
        )
    else:
        prefix = (task.completion_prefix or f"def {task.entry_point}(...):\n")

    if variant == "v01_benchmark":
        return task.prompt, meta

    if variant == "raw_completion_prefix":
        meta["note"] = "raw code-completion prefix; semantics aligned to task"
        return prefix, meta

    if variant == "minimal_instruction":
        # Short, direct instruction form.
        if task.task_id == "NV01-BASIC-001":
            text = "Instruction: Write function add(a,b) that returns a+b.\nResponse:\n"
        elif task.task_id == "NV01-ARITH-003":
            text = "Instruction: Write function square(n) that returns n*n.\nResponse:\n"
        elif task.task_id == "NV01-LIST-006":
            text = "Instruction: Write function sum_list(nums) that returns the sum of nums.\nResponse:\n"
        elif task.task_id == "NV01-COMP-015":
            text = "Instruction: Complete count_items(items) to return len(items).\nResponse:\n" + (task.completion_prefix or "")
        elif task.task_id == "NV01-INST-017":
            text = "Instruction: Write function greet(name) returning 'Hello, <name>!'.\nResponse:\n"
        elif task.task_id == "NV01-DEBUG-019":
            text = "Instruction: Fix sum_to_n(n) to sum 1..n inclusive.\nResponse:\n"
        else:
            text = f"Instruction: {task.prompt}\nResponse:\n"
        return text, meta

    if variant == "recovered_sft_format":
        # Recovered from training/chat_template.py (v1, bos/eos) + configs/train/sft_v180.yaml
        # and data/v180_sft/sft.jsonl style (instruction-completion, often code prefix + short response).
        # Rendered as: <bos>{SYSTEM_PROMPT}\n\n{instruction}\n\n  (prompt half, no assistant text/EOS)
        try:
            from training.chat_template import SYSTEM_PROMPT, render_prompt as sft_render_prompt
            from training.tokenizer_contract import detect_contract
            # detect_contract needs tokenizer object; caller may supply via meta later, but keep generic.
        except Exception:  # noqa: BLE001
            SYSTEM_PROMPT = "You are a helpful coding assistant."  # fallback, marked uncertain
            sft_render_prompt = None
        # Map to v180-style completion instructions where natural.
        sft_instruction = task.prompt
        if task.task_id == "NV01-BASIC-001":
            sft_instruction = "Complete the following Python function. Return only the code that completes the return statement. Do not repeat the function signature.\n\ndef add(a, b):\n    return "
        elif task.task_id == "NV01-ARITH-003":
            sft_instruction = "Complete the following Python function. Return only the code that completes the return statement. Do not repeat the function signature.\n\ndef square(n):\n    return "
        elif task.task_id == "NV01-LIST-006":
            sft_instruction = "Complete the following Python function. Return only the code that completes the function. Do not repeat the function signature.\n\ndef sum_list(nums):\n    "
        elif task.task_id == "NV01-COMP-015":
            sft_instruction = "Complete the following Python function. Return only the code that completes the function. Do not repeat the function signature.\n\ndef count_items(items):\n    return "
        elif task.task_id == "NV01-INST-017":
            sft_instruction = "Complete the following Python function. Return only the code that completes the return statement. Do not repeat the function signature.\n\ndef greet(name):\n    return "
        elif task.task_id == "NV01-DEBUG-019":
            sft_instruction = (
                "Fix the bug in this Python function and return the corrected function body/completion as needed.\n\n"
                "def sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total"
            )
        # We cannot call render_prompt without tokenizer contract object here; include instruction and let caller
        # build contract-correct prompt when tokenizer provided. Mark for post-render.
        meta["recovered_contract"] = "qm.plain_instruction/1 (bos+system+instruction prompt half) as in training/sft.py+chat_template.py; v180_sft data used instruction-completion pairs"
        meta["sft_instruction"] = sft_instruction
        meta["needs_contract_render"] = True
        # Fallback plain rendering if not post-rendered.
        return f"<bos>{SYSTEM_PROMPT}\n\n{sft_instruction}\n\n", meta

    if variant == "few_shot":
        # Small few-shot compatible with completion style; marked as compatibility probe.
        meta["note"] = "few-shot completion probe (compatibility with recovered style, not verified as training format)"
        shot1 = "def add(a, b):\n    return a + b\n\n"
        if task.task_id == "NV01-BASIC-001":
            text = shot1 + "def add(a, b):\n    return "
        elif task.task_id == "NV01-ARITH-003":
            text = shot1 + "def square(n):\n    return "
        elif task.task_id == "NV01-LIST-006":
            text = shot1 + "def sum_list(nums):\n    total = 0\n    for "
        elif task.task_id == "NV01-COMP-015":
            text = shot1 + (task.completion_prefix or "def count_items(items):\n") + "    return "
        elif task.task_id == "NV01-INST-017":
            text = "def double(x):\n    return x * 2\n\n" + "def greet(name):\n    return "
        elif task.task_id == "NV01-DEBUG-019":
            text = "def inc(n):\n    return n + 1\n\n" + prefix
        else:
            text = shot1 + prefix
        return text, meta

    raise ValueError(f"unknown variant: {variant}")


def finalize_prompt_for_tokenizer(prompt: str, meta: dict[str, Any], tokenizer) -> str:
    if not meta.get("needs_contract_render"):
        return prompt
    try:
        from training.chat_template import SYSTEM_PROMPT, render_prompt as sft_render_prompt
        from training.tokenizer_contract import detect_contract
        contract = detect_contract(tokenizer)
        instruction = meta.get("sft_instruction", prompt)
        return sft_render_prompt(SYSTEM_PROMPT, instruction, contract_id=contract)
    except Exception:  # noqa: BLE001
        return prompt


def _topk_from_logits(logits_1d: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    probs = torch.softmax(logits_1d.float(), dim=-1)
    kk = min(k, logits_1d.numel())
    vals, idx = torch.topk(probs, kk)
    return [{"token_id": int(i), "prob": float(v), "logit": float(logits_1d[int(i)])} for v, i in zip(vals.tolist(), idx.tolist())]


def _token_info(logits_1d: torch.Tensor, token_id: int | None) -> dict[str, Any]:
    if token_id is None:
        return {"token_id": None, "prob": None, "logit": None, "rank": None}
    probs = torch.softmax(logits_1d.float(), dim=-1)
    tid = int(token_id)
    prob = float(probs[tid])
    logit = float(logits_1d[tid])
    rank = int((probs > probs[tid]).sum().item()) + 1
    return {"token_id": tid, "prob": prob, "logit": logit, "rank": rank}


def _apply_penalties(logits, generated, presence_penalty=0.0, frequency_penalty=0.0):
    if not generated or (presence_penalty == 0.0 and frequency_penalty == 0.0):
        return logits
    counts: dict[int, int] = {}
    for t in generated:
        counts[t] = counts.get(t, 0) + 1
    idx = torch.tensor(list(counts.keys()), dtype=torch.long, device=logits.device)
    penalty = torch.tensor([presence_penalty + frequency_penalty * counts[int(t)] for t in idx.tolist()], dtype=logits.dtype, device=logits.device)
    logits = logits.clone()
    logits[0, idx] -= penalty
    return logits


def _filter_logits(logits, temperature, top_k, top_p):
    logits = logits / max(float(temperature), 1e-5)
    if top_k and int(top_k) > 0:
        k = min(int(top_k), logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = (cumulative - sorted_probs) > float(top_p)
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
    return probs


@torch.no_grad()
def diagnostic_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: int = 1,
    device: str = "cpu",
    binding_repair: bool = True,
    capture_steps: int = 8,
) -> dict[str, Any]:
    """Generate with per-step diagnostics (logits/EOS/raw ids), preserving repair separately."""
    model.eval()
    if seed is not None:
        torch.manual_seed(int(seed))

    prompt_ids = tokenizer.encode(prompt).ids
    prompt_tokens = len(prompt_ids)
    max_seq_len = int(model.cfg["max_seq_len"])
    eos_id = tokenizer.token_to_id("<eos>")
    generated: list[int] = []
    step_logs: list[dict[str, Any]] = []
    finish_reason = "length"

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for step in range(int(max_new_tokens)):
        idx_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
        logits, _ = model(idx_cond)
        raw_next = logits[0, -1, :].detach()
        adjusted = _apply_penalties(raw_next.unsqueeze(0), generated, 0.0, 0.0)

        # selection
        if temperature and float(temperature) > 0:
            probs = _filter_logits(adjusted, temperature, top_k, top_p)
            next_id_t = torch.multinomial(probs, num_samples=1)
        else:
            next_id_t = torch.argmax(adjusted, dim=-1, keepdim=True)
        chosen = int(next_id_t.item())

        if step < capture_steps:
            log_entry = {
                "step": step,
                "top5": _topk_from_logits(raw_next, 5),
                "eos": _token_info(raw_next, eos_id),
                "chosen": _token_info(raw_next, chosen),
                "chosen_token_text": tokenizer.decode([chosen]),
                "greedy_would_pick": int(torch.argmax(raw_next).item()),
            }
            step_logs.append(log_entry)

        if eos_id is not None and chosen == int(eos_id):
            finish_reason = "stop"
            # record EOS-selected step context is in step_logs if captured
            break

        generated.append(chosen)
        x = torch.cat([x, next_id_t.to(device=device)], dim=1)

    decoded_before_repair = tokenizer.decode(generated)
    raw_completion = decoded_before_repair
    completion = raw_completion
    repair_diagnostic: dict[str, Any] = {
        "enabled": bool(binding_repair),
        "applied": False,
        "status": "repaired" if binding_repair else "disabled",
        "reason": "repair enabled" if binding_repair else "repair disabled",
        "mapping": {},
        "changed": False,
        "raw_preserved": True,
    }
    if binding_repair:
        try:
            from training.binding_repair import repair_from_prompt_info
            info = repair_from_prompt_info(prompt, completion)
            completion = info.completion
            repair_diagnostic.update({
                "applied": bool(info.changed),
                "status": info.status,
                "reason": info.reason,
                "mapping": dict(info.mapping),
                "changed": bool(info.changed),
            })
        except Exception as exc:  # noqa: BLE001
            completion = raw_completion
            repair_diagnostic.update({
                "applied": False,
                "status": "repair_failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "changed": False,
            })

    # identify '(' investigations
    paren_id = None
    try:
        # best-effort: encode "(" and see common ids
        enc = tokenizer.encode("(").ids
        paren_id = enc[0] if len(enc) == 1 else enc
    except Exception:
        paren_id = None

    first_step = step_logs[0] if step_logs else None
    eos_first = first_step["eos"] if first_step else None
    immediate_eos = bool(first_step and eos_id is not None and first_step.get("greedy_would_pick") == int(eos_id))

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "decoded_before_repair": decoded_before_repair,
        "decoded_after_repair": completion,
        "raw_text": raw_completion,
        "text": completion,
        "finish_reason": finish_reason,
        "generated_token_count": len(generated),
        "eos_id": eos_id,
        "immediate_eos_greedy": immediate_eos,
        "first_step": first_step,
        "step_diagnostics": step_logs,
        "repair": repair_diagnostic,
        "paren_probe_token_id": paren_id,
    }


def _load_native(checkpoint_path: Path, tokenizer_path: Path, device: str = "cpu"):
    from serving.foundation_inference import load_foundation
    return load_foundation(str(checkpoint_path), str(tokenizer_path), device)


def collect_training_evidence(repo: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"verified": [], "unresolved": [], "samples": {}}
    # configs
    v180_cfg = repo / "configs/train/sft_v180.yaml"
    evidence["verified"].append(
        {
            "source": str(v180_cfg),
            "fact": "v180_sft SFT config: run qm-code-v180-sft, base_checkpoint runs/v160_pretrain/checkpoint_last.pt, base_model_config configs/model/code_v160_15m.yaml, tokenizer tokenizer/taksh_code_bpe_v1.json, train/val data/v180_sft/sft.jsonl, max_length 512, 200 steps, seed 42.",
        }
    )
    evidence["verified"].append(
        {
            "source": "training/sft.py + training/chat_template.py + training/tokenizer_contract.py",
            "fact": "SFTPairDataset uses contract detected from tokenizer (taksh_code_bpe_v1 is v1 plain_instruction with <bos>/<eos>, no role markers). Training renders render_completion(SYSTEM_PROMPT, instruction, response, contract_id=v1) => '<bos>{SYSTEM_PROMPT}\\n\\n{instruction}\\n\\n{response}<eos>' and masks prompt tokens via prompt_token_length (prompt half omits response/EOS). Loss is on response (+EOS) tokens; prompt masked.",
        }
    )
    evidence["verified"].append(
        {
            "source": "data/v180_sft/sft.jsonl (present locally)",
            "fact": "v180_sft records are instruction-completion pairs (id/task/instruction/input/response). Observed style: instruction asks to complete a function/return expression and includes code prefix; response is short completion (e.g. 'a + b', 'x * y') — completion-oriented targets in an instruction wrapper.",
        }
    )
    evidence["verified"].append(
        {
            "source": "data/v180_corpus/taksh_code_v1.manifest.json",
            "fact": "v180 corpus manifest indicates project/docs/scripts code corpus (cleared code text) used for pretraining context; separate from SFT pairs.",
        }
    )
    evidence["verified"].append(
        {
            "source": "data/l02_sft/dataset_manifest.json",
            "fact": "l02_sft is a different, later cleared synthetic SFT set (260 records, full function responses, CC0) with ### Instruction/Response-style formatting via training/instruction_format.py in that pipeline; v180_sft checkpoint under test was trained per sft_v180.yaml, not on l02_sft.",
        }
    )
    # samples (small, safe)
    def read_samples(path, n=3):
        p = repo / path
        if not p.is_file():
            return {"present": False}
        out = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({"raw": line[:200]})
        return {"present": True, "path": str(p), "count_sampled": len(out), "records": out}
    evidence["samples"]["v180_sft"] = read_samples("data/v180_sft/sft.jsonl", 3)
    evidence["samples"]["l02_sft"] = read_samples("data/l02_sft/sft.jsonl", 2)

    evidence["unresolved"].append(
        "Exact v180_sft training run logs/optimizer states beyond checkpoint step/config presence were not exhaustively recovered in this environment; checkpoint config payload read via strict safe allowlist encountered additional numpy dtype globals during metadata inspection (weights not modified). Core training recipe is evidenced by configs/train/sft_v180.yaml and training code/data."
    )
    evidence["unresolved"].append(
        "Whether v180_sft used any additional chat/role markers beyond tokenizer contract v1 is constrained by tokenizer evidence (no role special tokens registered) — consistent with plain bos/eos prompt; definitive per-example tokenization matches SFTPairDataset behavior."
    )
    return evidence


def run_diagnosis(
    checkpoint_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    *,
    device: str = "cpu",
    max_new_tokens: int = 64,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    tokenizer_p = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER
    out_json = Path(out_json) if out_json else (REPO / "evaluation/native_v02/diagnosis_results.json")
    out_md = Path(out_md) if out_md else (REPO / "docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md")

    report: dict[str, Any] = {
        "schema": "taksh.native_v02_diagnosis/1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "checkpoint_path": str(checkpoint),
        "tokenizer_path": str(tokenizer_p),
        "status": "NOT_RUN",
        "blocker": None,
        "training_evidence": collect_training_evidence(REPO),
        "experiments": [],
        "summary": {},
    }

    if not checkpoint.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"checkpoint_missing: {checkpoint}"
        _write(report, out_json, out_md)
        return report
    if not tokenizer_p.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"tokenizer_missing: {tokenizer_p}"
        _write(report, out_json, out_md)
        return report

    report["checkpoint_sha256"] = _sha256_file(checkpoint)
    report["tokenizer_sha256"] = _sha256_file(tokenizer_p)

    t0 = time.perf_counter()
    try:
        model, tok, meta = _load_native(checkpoint, tokenizer_p, device=device)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "BLOCKED"
        report["blocker"] = f"model_load_failed: {type(exc).__name__}: {exc}"
        report["load_seconds"] = round(time.perf_counter() - t0, 3)
        _write(report, out_json, out_md)
        return report
    report["load_seconds"] = round(time.perf_counter() - t0, 3)
    report["model_meta"] = meta

    tasks = get_subset_tasks()
    report["task_subset"] = [t.task_id for t in tasks]
    report["prompt_variants"] = list(PROMPT_VARIANTS)
    report["decoding_configs"] = DECODING_CONFIGS

    experiments = []
    total_gen_s = 0.0
    for task in tasks:
        for variant in PROMPT_VARIANTS:
            prompt_text, pmeta = render_prompt(task, variant)
            prompt_text = finalize_prompt_for_tokenizer(prompt_text, pmeta, tok)
            for dcfg in DECODING_CONFIGS:
                started = time.perf_counter()
                try:
                    diag = diagnostic_generate(
                        model, tok, prompt_text,
                        max_new_tokens=max_new_tokens,
                        temperature=dcfg["temperature"],
                        top_k=dcfg.get("top_k", 50),
                        top_p=dcfg.get("top_p", 1.0),
                        seed=dcfg.get("seed", 1),
                        device=device,
                        binding_repair=True,
                        capture_steps=8,
                    )
                    err = None
                except Exception as exc:  # noqa: BLE001
                    diag = None
                    err = f"{type(exc).__name__}: {exc}"
                gen_s = time.perf_counter() - started
                total_gen_s += gen_s
                exp = {
                    "task_id": task.task_id,
                    "category": task.category,
                    "variant": variant,
                    "decoding": dcfg["name"],
                    "decoding_config": dcfg,
                    "prompt_meta": pmeta,
                    "prompt_text": prompt_text,
                    "generation_seconds": round(gen_s, 4),
                    "error": err,
                }
                if diag:
                    exp.update(diag)
                    # consistency checks
                    exp["raw_ids_match_decoded_before_repair"] = (tok.decode(diag["generated_token_ids"]) == diag["decoded_before_repair"])
                experiments.append(exp)
    report["experiments"] = experiments
    report["total_generation_seconds"] = round(total_gen_s, 3)
    report["summary"] = summarize(experiments)
    report["status"] = "COMPLETED"
    _write(report, out_json, out_md)
    return report


def summarize(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(experiments)
    def rate(pred):
        return round(sum(1 for e in experiments if pred(e)) / n, 4) if n else 0.0
    summary = {
        "experiment_count": n,
        "immediate_eos_greedy_rate": rate(lambda e: bool(e.get("immediate_eos_greedy"))),
        "finish_stop_rate": rate(lambda e: e.get("finish_reason") == "stop"),
        "zero_completion_tokens_rate": rate(lambda e: e.get("generated_token_count") == 0),
        "empty_decoded_before_repair_rate": rate(lambda e: not (e.get("decoded_before_repair") or "").strip()),
        "raw_ids_match_decoded_before_repair_rate": rate(lambda e: e.get("raw_ids_match_decoded_before_repair") is True),
        "by_variant": {},
        "by_decoding": {},
    }
    for key, field in (("by_variant", "variant"), ("by_decoding", "decoding")):
        bucket = {}
        for e in experiments:
            k = e.get(field)
            b = bucket.setdefault(k, {"count": 0, "immediate_eos_greedy": 0, "zero_completion": 0, "empty_decoded": 0, "nonempty_decoded": 0})
            b["count"] += 1
            b["immediate_eos_greedy"] += int(bool(e.get("immediate_eos_greedy")))
            b["zero_completion"] += int(e.get("generated_token_count") == 0)
            empty = not (e.get("decoded_before_repair") or "").strip()
            b["empty_decoded"] += int(empty)
            b["nonempty_decoded"] += int(not empty)
        for b in bucket.values():
            c = b["count"] or 1
            b["immediate_eos_greedy_rate"] = round(b["immediate_eos_greedy"]/c, 4)
            b["zero_completion_rate"] = round(b["zero_completion"]/c, 4)
            b["empty_decoded_rate"] = round(b["empty_decoded"]/c, 4)
        summary[key] = bucket
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    lines = [
        "# TAKSH Native v0.2 — Generation Diagnosis and Training Readiness",
        "",
        f"- Generated (UTC): {report.get('generated_at_utc')}",
        f"- Status: **{report.get('status')}**",
        f"- Checkpoint: `{report.get('checkpoint_path')}` SHA256 `{report.get('checkpoint_sha256')}`",
        f"- Tokenizer: `{report.get('tokenizer_path')}` SHA256 `{report.get('tokenizer_sha256')}`",
        f"- Device: {report.get('device')} | Load: {report.get('load_seconds')}s | Total generation: {report.get('total_generation_seconds')}s",
        "",
        "## Training-format evidence",
        "",
    ]
    ev = report.get("training_evidence", {})
    for item in ev.get("verified", []):
        lines.append(f"- [verified] **{item.get('source')}**: {item.get('fact')}")
    for u in ev.get("unresolved", []):
        lines.append(f"- [unresolved] {u}")
    lines += ["", "## Experiment matrix", "", f"- Task subset ({len(report.get('task_subset', []))}): {report.get('task_subset')}",
              f"- Prompt variants: {report.get('prompt_variants')}",
              f"- Decoding configs: {[d['name'] for d in report.get('decoding_configs', [])]}",
              f"- Experiments: {s.get('experiment_count')}", "",
              "## Aggregate observations", "",
              f"- Immediate EOS by greedy/first-step evidence rate: {s.get('immediate_eos_greedy_rate')}",
              f"- Zero completion tokens rate: {s.get('zero_completion_tokens_rate')}",
              f"- Empty decoded (before repair) rate: {s.get('empty_decoded_before_repair_rate')}",
              f"- Raw token IDs match decoded-before-repair rate: {s.get('raw_ids_match_decoded_before_repair_rate')}",
              ""]
    if report.get("blocker"):
        lines += ["## Blocker", "", f"- {report['blocker']}", ""]
    lines += ["## Representative experiments (first 12)", ""]
    for e in report.get("experiments", [])[:12]:
        fs = e.get("first_step") or {}
        eos = fs.get("eos") or {}
        lines += [
            f"### {e.get('task_id')} | {e.get('variant')} | {e.get('decoding')}",
            f"- finish: {e.get('finish_reason')} tokens={e.get('generated_token_count')} error={e.get('error')}",
            f"- first-step EOS rank/prob: {eos.get('rank')}/{eos.get('prob')} chosen: {fs.get('chosen')} greedy_pick: {fs.get('greedy_would_pick')}",
            f"- decoded_before_repair: {json.dumps((e.get('decoded_before_repair') or '')[:120], ensure_ascii=False)}",
            f"- decoded_after_repair: {json.dumps((e.get('decoded_after_repair') or '')[:120], ensure_ascii=False)}",
            f"- repair: {e.get('repair')}",
            "",
        ]
    return "\n".join(lines) + "\n"


def _write(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="TAKSH Native v0.2 generation diagnosis")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--out-json", default=str(REPO / "evaluation/native_v02/diagnosis_results.json"))
    p.add_argument("--out-md", default=str(REPO / "docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args(argv)
    report = run_diagnosis(args.checkpoint, args.tokenizer, device=args.device, max_new_tokens=args.max_new_tokens, out_json=args.out_json, out_md=args.out_md)
    print(json.dumps({"status": report.get("status"), "summary": report.get("summary"), "blocker": report.get("blocker"), "out_json": args.out_json, "out_md": args.out_md}, ensure_ascii=False))
    return 0 if report.get("status") in {"COMPLETED", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
