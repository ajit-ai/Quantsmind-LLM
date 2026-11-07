"""M3 hardware and feasibility preflight.

Run this before a training run that costs real time. It answers one question:
**will this configuration finish on this machine?**

It deliberately separates two kinds of number:

``measured``
    read from the running system, or observed by actually executing a few
    optimizer steps. Peak memory and throughput are measured, not modelled.

``estimated``
    arithmetic from the config, clearly labelled, never presented as fact.

The verdict is one of:

``FEASIBLE``
    The configuration fits with a safety margin.
``FEASIBLE_WITH_SMALLER_CONFIG``
    It does not fit, but a specific, named reduction would. The suggestion is
    printed; nothing is changed automatically.
``BLOCKED``
    A gate that code must not bypass: unresolved licences, missing or
    unverifiable data, tokenizer/vocabulary mismatch, missing artifacts.

The ``--override`` flag acknowledges *resource* warnings only. It can never
clear a BLOCKED gate, because a memory warning and "this data is not licensed
for training" are not the same kind of fact.

Usage
-----
    python -m training.preflight --config configs/train/pretrain_m3.yaml
    python -m training.preflight --config ... --probe-steps 3 --json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

FEASIBLE = "FEASIBLE"
FEASIBLE_SMALLER = "FEASIBLE_WITH_SMALLER_CONFIG"
BLOCKED = "BLOCKED"

# Never plan to consume the whole machine: training that leaves no headroom
# gets OOM-killed by the OS rather than failing cleanly.
DEFAULT_MEMORY_MARGIN = 0.30

# fp32 AdamW: param + grad + exp_avg + exp_avg_sq.
BYTES_PER_PARAM_ADAMW_FP32 = 16


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory():
    """Total and available RAM in bytes. Measured, not estimated."""
    total = available = None
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
        available = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - fall through to platform APIs
        pass

    if total is None and platform.system() == "Windows":
        try:
            status = _MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass

    if total is None and Path("/proc/meminfo").exists():
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
        except Exception:  # noqa: BLE001
            pass

    return {"total_bytes": total, "available_bytes": available, "kind": "measured"}


def torch_facts():
    import torch

    facts = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "torch_threads": torch.get_num_threads(),
    }
    if facts["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        facts["cuda_device_name"] = torch.cuda.get_device_name(0)
        facts["cuda_total_memory_bytes"] = int(props.total_memory)
        facts["cuda_kind"] = "measured"
    else:
        facts["cuda_device_name"] = None
        facts["cuda_total_memory_bytes"] = None
        facts["cuda_kind"] = "absent"
    return facts


def disk_space(path="."):
    usage = shutil.disk_usage(str(Path(path)))
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "kind": "measured",
    }


def peak_rss_bytes():
    """Peak resident set size of this process, where the OS reports it.

    Linux/macOS via ``resource``; Windows via ``GetProcessMemoryInfo``
    (``PeakWorkingSetSize``). Returns ``None`` only when the platform offers
    neither, in which case callers fall back to the estimate and say so.
    """
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak * 1024 if platform.system() != "Darwin" else peak
    except Exception:  # noqa: BLE001 - not available on Windows
        pass

    if platform.system() == "Windows":
        try:
            import ctypes.wintypes as wt

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)

            get_current = ctypes.windll.kernel32.GetCurrentProcess
            get_current.restype = wt.HANDLE
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            # Without explicit argtypes ctypes marshals the pseudo-handle -1 and
            # the struct pointer incorrectly, and the call simply returns 0.
            query.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wt.DWORD]
            query.restype = wt.BOOL

            if query(get_current(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # noqa: BLE001
            pass
    return None


def load_configs(train_config_path):
    import yaml

    train_path = Path(train_config_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train config not found: {train_path}")
    train_cfg = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}

    model_cfg_path = (train_cfg.get("run") or {}).get("model_config")
    if not model_cfg_path:
        raise ValueError(f"{train_path}: missing run.model_config")
    loaded = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8")) or {}
    return train_cfg, loaded.get("model", loaded), Path(model_cfg_path)


def resolve_tokens(train_cfg):
    data_cfg = train_cfg.get("data") or {}
    if data_cfg.get("tokens_path"):
        return Path(data_cfg["tokens_path"])
    tag = data_cfg.get("dataset_tag")
    if tag:
        return Path("data") / f"{tag.replace('-', '_')}.tokens.bin"
    raise ValueError(
        "cannot resolve train tokens: set data.tokens_path or data.dataset_tag"
    )


def resolve_val_tokens(train_cfg):
    data_cfg = train_cfg.get("data") or {}
    return Path(data_cfg["val_tokens_path"]) if data_cfg.get("val_tokens_path") else None


# --------------------------------------------------------------------------- #
# Gates that must never be overridable
# --------------------------------------------------------------------------- #
def check_hard_gates(train_cfg, tokens_path, val_path, model_cfg):
    """Licence, data-integrity and tokenizer gates. ``--override`` cannot clear these.

    A memory warning and "this data is not licensed for training" are not the
    same kind of fact, so resource pressure is handled elsewhere.
    """
    from training.data import TokenDataset

    gates = []

    def record(name, ok, detail, remedy=None):
        gates.append({"gate": name, "ok": bool(ok), "detail": detail,
                      "remedy": remedy})
        return ok

    tokens_ok = Path(tokens_path).exists()
    record(
        "train_artifact_present", tokens_ok,
        f"{tokens_path} found" if tokens_ok else f"missing: {tokens_path}",
        None if tokens_ok else "python -m data.prepare_m3_dataset",
    )

    if val_path is None:
        record("val_artifact_present", True, "no validation artifact configured")
    else:
        val_ok = Path(val_path).exists()
        record(
            "val_artifact_present", val_ok,
            f"{val_path} found" if val_ok else f"missing: {val_path}",
            None if val_ok else "python -m data.prepare_m3_dataset",
        )

    # Licence evidence travels with the dataset manifest, not with the token file.
    manifest_path = (train_cfg.get("data") or {}).get("dataset_manifest")
    if not manifest_path or not Path(manifest_path).exists():
        record(
            "licence_evidence_present", False,
            f"dataset manifest missing: {manifest_path}",
            "rebuild with python -m data.prepare_m3_dataset",
        )
    else:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        gate = manifest.get("licence_gate") or {}
        withheld = gate.get("withheld_records")
        unresolved_allowed = gate.get("unreviewed_allowed")
        ok = withheld is not None and not unresolved_allowed
        record(
            "licence_evidence_present", ok,
            f"verified sources={len(gate.get('verified_sources') or [])}, "
            f"withheld records={withheld}, unreviewed_allowed={unresolved_allowed}",
            None if ok else (
                "dataset contains sources with unresolved licences; rebuild "
                "with --allow-unreviewed-licences only after a recorded human decision"
            ),
        )

    # Tokenizer identity and vocabulary width must agree with the model.
    tokenizer_path = (train_cfg.get("data") or {}).get("tokenizer")
    model_vocab = int(model_cfg.get("vocab_size") or 0)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        record(
            "tokenizer_present", False,
            f"tokenizer missing: {tokenizer_path}",
            "set data.tokenizer in the train config",
        )
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        tok_vocab = tok.get_vocab_size()
        record("tokenizer_present", True, f"{tokenizer_path} vocab={tok_vocab}")
        record(
            "vocab_matches_model", model_vocab == tok_vocab,
            f"model vocab_size={model_vocab}, tokenizer vocab={tok_vocab}",
            None if model_vocab == tok_vocab else (
                "the model vocabulary must equal the tokenizer vocabulary; "
                "do not resize the tokenizer or truncate ids"
            ),
        )

    # The token artifact must itself be readable and agree with that tokenizer.
    if tokens_ok and tokenizer_path and Path(tokenizer_path).exists():
        try:
            dataset = TokenDataset(tokens_path)
            problems = dataset.validate_for_model(model_vocab)
            incompatible = [
                p for p in problems
                if "exceeds model vocab_size" in p or "does not match" in p
            ]
            record(
                "token_artifact_compatible", not incompatible,
                "; ".join(problems) if problems else "artifact agrees with tokenizer",
                None if not incompatible else incompatible[0],
            )
        except Exception as exc:  # noqa: BLE001 - reported as a failed gate
            record("token_artifact_compatible", False,
                   f"cannot read token artifact: {type(exc).__name__}: {exc}")

    return gates


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
def model_size(model_cfg):
    from training.model import QMModel

    model = QMModel(model_cfg)
    total = sum(p.numel() for p in model.parameters())
    del model
    return {
        "total_params": total,
        "non_embedding_params": None,  # filled by the caller via QMModel.num_params
    }


def estimate_memory(model_cfg, train_cfg):
    """Steady-state memory. Arithmetic from the config, labelled ESTIMATED.

    The probe in :func:`measure_probe` replaces this with a measured figure
    when it runs; this estimate exists so a verdict is possible without
    allocating the model at all.
    """
    from training.model import QMModel

    model = QMModel(model_cfg)
    params = sum(p.numel() for p in model.parameters())
    del model

    data_cfg = train_cfg.get("data") or {}
    batch_cfg = train_cfg.get("batch") or {}
    seq_len = int(data_cfg.get("seq_len") or 0)
    micro_bs = int(batch_cfg.get("micro_batch_size") or 1)

    precision = ((train_cfg.get("precision") or {}).get("dtype") or "float32").lower()
    param_bytes = {"float32": 4, "bfloat16": 2, "float16": 2}.get(precision, 4)
    # AdamW keeps two moment buffers in param dtype.
    steady = params * param_bytes * (2 if param_bytes == 4 else 3)
    moments = params * param_bytes * 2

    d_model = int(model_cfg.get("d_model") or 0)
    n_layers = int(model_cfg.get("n_layers") or 0)
    # Per-layer stored activations for the forward pass: attention projections
    # (4 * d_model), the SwiGLU hidden state (ffn_dim) and its saved inputs.
    ffn_dim = int(model_cfg.get("ffn_dim") or 0)
    per_token = 4 * d_model + 3 * ffn_dim
    activations = micro_bs * seq_len * per_token * n_layers * param_bytes

    return {
        "kind": "estimated",
        "params": params,
        "param_dtype_bytes": param_bytes,
        "weights_plus_grads_bytes": steady,
        "optimizer_moments_bytes": moments,
        "total_steady_state_bytes": steady + moments,
        "activation_bytes": activations,
        "total_bytes": steady + moments + activations,
        "basis": (
            "weights + grads + AdamW exp_avg/exp_avg_sq at param dtype, plus "
            "stored forward activations (4*d_model + 3*ffn_dim) per token per layer"
        ),
    }


def measure_probe(model_cfg, train_cfg, steps=3, device="cpu"):
    """Build the real model and run real optimizer steps. Everything here is measured.

    This is the difference between "should fit" and "fits": peak RSS after
    actual forward/backward passes, and actual tokens per second.
    """
    import numpy as np
    import torch

    from training.data import TokenDataset
    from training.model import QMModel
    from training.pretrain import lr_at

    data_cfg = train_cfg.get("data") or {}
    batch_cfg = train_cfg.get("batch") or {}
    seq_len = int(data_cfg.get("seq_len") or 0)
    micro_bs = int(batch_cfg.get("micro_batch_size") or 1)
    accum = int(batch_cfg.get("grad_accum_steps") or 1)

    tokens_path = resolve_tokens(train_cfg)
    dataset = TokenDataset(tokens_path)
    model = QMModel(model_cfg).to(device)
    model.train()
    opt_cfg = train_cfg.get("optimizer") or {}
    optim = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg.get("lr_peak", 3e-3),
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )

    rss_before = peak_rss_bytes()
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        lr = lr_at(step, steps, opt_cfg.get("lr_peak", 3e-3),
                   opt_cfg.get("warmup_steps", 0), opt_cfg.get("min_ratio", 0.1))
        for group in optim.param_groups:
            group["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(accum):
            x, y = dataset.batch(micro_bs, seq_len)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += float(loss.item()) / accum
        if opt_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
        optim.step()
        losses.append(loss_val)
    elapsed = time.perf_counter() - t0
    rss_after = peak_rss_bytes()

    tokens_per_step = micro_bs * accum * seq_len
    del model, optim, dataset
    return {
        "kind": "measured",
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": (tokens_per_step * steps / elapsed) if elapsed else None,
        "peak_rss_bytes": rss_after,
        "rss_before_bytes": rss_before,
        "rss_delta_bytes": (rss_after - rss_before)
        if (rss_after is not None and rss_before is not None) else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "losses_finite": all(np.isfinite(losses)) if losses else None,
        "device": device,
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def suggest_smaller(train_cfg, model_cfg, required_bytes, available_bytes):
    """Name a concrete reduction, without applying it."""
    suggestions = []
    layers = int(model_cfg.get("n_layers") or 0)
    d_model = int(model_cfg.get("d_model") or 0)
    shortfall = required_bytes - available_bytes
    suggestions.insert(0, (
        f"this configuration needs about {required_bytes / 2**20:.0f} MiB but "
        f"only {available_bytes / 2**20:.0f} MiB is available "
        f"(short by {shortfall / 2**20:.0f} MiB)"
    ))
    if layers > 2:
        suggestions.append(
            f"reduce n_layers {layers} -> {max(2, layers // 2)} (halves the "
            "dominant activation and parameter cost)"
        )
    if d_model > 64:
        suggestions.append(
            f"reduce d_model {d_model} -> {d_model // 2} (roughly quarters "
            "parameter-dependent memory)"
        )
    data_cfg = train_cfg.get("data") or {}
    seq_len = int(data_cfg.get("seq_len") or 0)
    if seq_len > 256:
        suggestions.append(
            f"reduce data.seq_len {seq_len} -> {seq_len // 2} (activations "
            "scale linearly in sequence length)"
        )
    return suggestions


def run_preflight(config_path, probe_steps=0, margin=DEFAULT_MEMORY_MARGIN,
                  override=False, disk_path="."):
    train_cfg, model_cfg, model_cfg_path = load_configs(config_path)
    tokens_path = resolve_tokens(train_cfg)
    val_path = resolve_val_tokens(train_cfg)

    gates = check_hard_gates(train_cfg, tokens_path, val_path, model_cfg)
    hard_failures = [g for g in gates if not g["ok"]]

    memory = system_memory()
    torch_info = torch_facts()
    disk = disk_space(disk_path)

    from training.model import QMModel
    probe_model = QMModel(model_cfg)
    params = sum(p.numel() for p in probe_model.parameters())
    non_embedding = probe_model.num_params()
    del probe_model

    estimate = estimate_memory(model_cfg, train_cfg)
    probe = (
        measure_probe(model_cfg, train_cfg, steps=probe_steps)
        if probe_steps > 0 and not hard_failures else None
    )

    data_cfg = train_cfg.get("data") or {}
    batch_cfg = train_cfg.get("batch") or {}
    seq_len = int(data_cfg.get("seq_len") or 0)
    micro_bs = int(batch_cfg.get("micro_batch_size") or 1)
    accum = int(batch_cfg.get("grad_accum_steps") or 1)
    max_steps = int((train_cfg.get("training") or {}).get("max_steps") or 0)
    tokens_per_step = micro_bs * accum * seq_len

    train_tokens = Path(tokens_path).stat().st_size // 2 if Path(tokens_path).exists() else None
    val_tokens = (
        Path(val_path).stat().st_size // 2
        if val_path and Path(val_path).exists() else None
    )

    # Prefer a measured peak over the arithmetic estimate.
    if probe and probe.get("rss_delta_bytes"):
        required = probe["rss_delta_bytes"]
        basis = "measured (peak RSS delta over real optimizer steps)"
    else:
        required = estimate["total_bytes"]
        basis = "estimated (no probe run)"

    available = memory["available_bytes"]
    total = memory["total_bytes"]
    suggestions = []
    budget = None
    total_budget = None
    fits = True
    under_pressure = False

    # Availability on a shared machine moves as other programs start and stop,
    # so it is not a stable property of this configuration. Two comparisons:
    #
    #   1. against currently-available RAM -> strict, but noisy
    #   2. against total RAM               -> stable, says whether the run can
    #                                         fit the machine at all
    #
    # The verdict is strict when (1) passes, permissive-with-warning when only
    # (2) passes, and rejects only when neither does.
    if available:
        budget = int(available * (1 - margin))
        fits = required <= budget
        if not fits:
            if total and required <= int(total * (1 - margin)):
                under_pressure = True
            else:
                suggestions = suggest_smaller(train_cfg, model_cfg, required, budget)
    if total:
        total_budget = int(total * (1 - margin))

    runtime_estimate = None
    if probe and probe.get("tokens_per_second"):
        runtime_estimate = (max_steps * tokens_per_step) / probe["tokens_per_second"]

    warnings = []
    if not torch_info["cuda_available"]:
        warnings.append("no CUDA device present: training is CPU-only on this host")
    if torch_info["torch_threads"] and torch_info["torch_threads"] < 4:
        warnings.append(f"only {torch_info['torch_threads']} torch threads")
    if disk["free_bytes"] < 2 * 2**30:
        warnings.append(f"only {disk['free_bytes'] / 2**30:.1f} GiB free disk")
    if under_pressure:
        warnings.append(
            f"host memory pressure: {available / 2**20:.0f} MiB available right "
            f"now (other programs are using the machine), but this run needs "
            f"{required / 2**20:.0f} MiB of the {total / 2**30:.2f} GiB total. "
            "The run fits the machine; it will start more reliably if memory is "
            "freed first."
        )

    if hard_failures:
        verdict = BLOCKED
        headline = (
            f"BLOCKED: {len(hard_failures)} gate(s) that must not be overridden "
            "have failed."
        )
    elif not fits and not under_pressure:
        verdict = FEASIBLE_SMALLER
        headline = (
            f"Configuration does not fit with a {margin:.0%} safety margin; a "
            "smaller configuration is available."
        )
    elif under_pressure:
        verdict = FEASIBLE
        headline = (
            "Configuration fits the machine, but available RAM is currently "
            "reduced by other programs. See warnings."
        )
    else:
        verdict = FEASIBLE
        headline = "Configuration fits with the required safety margin."

    return {
        "schema": "quantsmind.m3_preflight/1",
        "generated_at": utc_now_iso(),
        "verdict": verdict,
        "headline": headline,
        "config": str(Path(config_path)),
        "model_config": str(model_cfg_path),
        "override_acknowledged": bool(override),
        "override_note": (
            "--override acknowledges resource warnings only; it cannot clear "
            "a BLOCKED gate"
        ),
        "gates": gates,
        "hard_gate_failures": hard_failures,
        "model": {
            "params": params,
            "non_embedding_params": non_embedding,
            "vocab_size": model_cfg.get("vocab_size"),
            "d_model": model_cfg.get("d_model"),
            "n_layers": model_cfg.get("n_layers"),
            "max_seq_len": model_cfg.get("max_seq_len"),
        },
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "memory": memory,
            "disk": disk,
            "torch": torch_info,
        },
        "data": {
            "train_tokens_path": str(tokens_path),
            "train_tokens": train_tokens,
            "val_tokens_path": str(val_path) if val_path else None,
            "val_tokens": val_tokens,
            "seq_len": seq_len,
            "epochs_estimate": (
                (max_steps * tokens_per_step / train_tokens) if train_tokens else None
            ),
        },
        "schedule": {
            "max_steps": max_steps,
            "micro_batch_size": micro_bs,
            "grad_accum_steps": accum,
            "tokens_per_step": tokens_per_step,
            "total_tokens": max_steps * tokens_per_step,
        },
        "memory_estimate": estimate,
        "memory_required_bytes": required,
        "memory_required_basis": basis,
        "memory_budget_bytes": budget,
        "memory_total_budget_bytes": total_budget,
        "host_memory_pressure": under_pressure,
        "memory_margin_basis": (
            "compared against currently-available RAM when possible, else "
            "against total RAM; availability moves with other programs"
        ),
        "safety_margin": margin,
        "probe": probe,
        "runtime_estimate_seconds": runtime_estimate,
        "runtime_basis": (
            "measured throughput x planned tokens" if runtime_estimate else None
        ),
        "warnings": warnings,
        "suggestions": suggestions,
    }


def format_report(report) -> str:
    lines = []
    add = lines.append
    add("M3 preflight")
    add(f"  config : {report['config']}")
    add(f"  verdict: {report['verdict']}")
    add(f"  {report['headline']}")
    add("")

    add("  Gates (must not be overridden):")
    for gate in report["gates"]:
        mark = "PASS" if gate["ok"] else "FAIL"
        add(f"    [{mark}] {gate['gate']}: {gate['detail']}")
        if gate.get("remedy"):
            add(f"           remedy: {gate['remedy']}")
    add("")

    system = report["system"]
    add("  System (measured):")
    add(f"    python {system['python']} | torch {system['torch']['torch_version']}")
    mem = system["memory"]
    if mem["total_bytes"]:
        add(f"    RAM total {mem['total_bytes'] / 2**30:.2f} GiB, "
            f"available {mem['available_bytes'] / 2**30:.2f} GiB")
    add(f"    CUDA: {system['torch']['cuda_device_name'] or 'absent'} | "
        f"threads {system['torch']['torch_threads']} | "
        f"logical CPUs {system['logical_cpus']}")
    add(f"    disk free {system['disk']['free_bytes'] / 2**30:.1f} GiB")
    add("")

    model = report["model"]
    add("  Model:")
    add(f"    params {model['params']:,} | vocab {model['vocab_size']} | "
        f"d_model {model['d_model']} | layers {model['n_layers']}")
    add("")

    schedule = report["schedule"]
    data = report["data"]
    add("  Schedule:")
    add(f"    {schedule['max_steps']} steps x {schedule['tokens_per_step']} "
        f"tokens/step = {schedule['total_tokens']:,} tokens")
    if data["train_tokens"]:
        epochs = (f", ~{data['epochs_estimate']:.2f} epochs"
                  if data["epochs_estimate"] else "")
        add(f"    train stream {data['train_tokens']:,} tokens{epochs}")
    if data["val_tokens"]:
        add(f"    val stream   {data['val_tokens']:,} tokens (evaluation only)")
    add("")

    add("  Memory:")
    add(f"    required {report['memory_required_bytes'] / 2**20:.0f} MiB "
        f"({report['memory_required_basis']})")
    if report["memory_budget_bytes"]:
        add(f"    budget   {report['memory_budget_bytes'] / 2**20:.0f} MiB "
            f"({report['safety_margin']:.0%} safety margin)")
    estimate = report["memory_estimate"]
    add(f"    estimated: steady-state "
        f"{estimate['total_steady_state_bytes'] / 2**20:.0f} MiB, activations "
        f"{estimate['activation_bytes'] / 2**20:.0f} MiB")
    add("")

    probe = report["probe"]
    if probe:
        add("  Probe (measured by running real optimizer steps):")
        add(f"    {probe['steps']} steps in {probe['elapsed_seconds']:.2f}s -> "
            f"{probe['tokens_per_second']:.0f} tokens/s")
        add(f"    loss {probe['first_loss']:.4f} -> {probe['last_loss']:.4f} "
            f"(finite={probe['losses_finite']})")
        if report["runtime_estimate_seconds"]:
            add(f"    projected runtime for {schedule['max_steps']} steps: "
                f"{report['runtime_estimate_seconds'] / 60:.1f} min "
                "(measured throughput x planned tokens)")
        add("")

    if report["warnings"]:
        add("  Warnings:")
        for warning in report["warnings"]:
            add(f"    - {warning}")
        add("")

    if report["suggestions"]:
        add("  Suggestions (nothing was changed automatically):")
        for suggestion in report["suggestions"]:
            add(f"    - {suggestion}")
        add("")

    if report["override_acknowledged"]:
        add(f"  note: {report['override_note']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M3 hardware and feasibility preflight"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--probe-steps", type=int, default=0,
        help="run this many real optimizer steps and measure peak memory "
             "and throughput (recommended before a long run)",
    )
    parser.add_argument("--margin", type=float, default=DEFAULT_MEMORY_MARGIN)
    parser.add_argument(
        "--override", action="store_true",
        help="acknowledge RESOURCE warnings. Cannot clear a BLOCKED gate.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", default=None)
    args = parser.parse_args(argv)

    try:
        report = run_preflight(
            args.config, probe_steps=args.probe_steps, margin=args.margin,
            override=args.override,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"preflight could not run: {exc}")
        return 2

    if args.write:
        Path(args.write).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    return {FEASIBLE: 0, FEASIBLE_SMALLER: 1, BLOCKED: 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())