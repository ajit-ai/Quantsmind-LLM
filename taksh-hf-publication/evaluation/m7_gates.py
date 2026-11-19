"""M7 gates A-H and the machine-readable authorization/evidence records.

The gates are deliberately separate from the code that trains and evaluates.
A gate report that is produced by the same function it is supposed to constrain
is not a gate.

Pre-training gates (A-D) can fail the milestone before any optimizer step.
Gate E is the explicit authorization record consumed by the training commands.
Post-training gates (F-H) either verify the run or refuse to turn a negative
result into a positive claim.

No gate silently passes: every gate records the exact evidence it used, and a
missing input is a failure, not an empty success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m5_schema import load as load_m5
from evaluation.m5_schema import manifest_hash as m5_manifest_hash
from evaluation.m7_evaluate import (
    load_jsonl,
    load_protocol,
    score_documentation,
    score_explanation,
    score_generation_syntax,
    score_json,
    score_sql,
    score_table,
    score_yaml,
    sha256_file,
)

SCHEMA = "quantsmind.m7_gates/1"
FIXTURE_SOURCE = "deterministic-fixture"
TRAIN_SOURCES = {
    "pallets-click", "pallets-flask", "pytest-dev-pytest",
    "tiangolo-fastapi", "urllib3-urllib3",
}
UNIFORM_LOSS = 8.669399124305569

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _partition_paths(dataset: dict) -> dict[str, str]:
    return {name: dataset[name] for name in ("train", "val", "dev", "locked")}


def _verified_sources(manifest: dict) -> set[str]:
    return set(manifest["licence_gate"]["verified_sources"])


def gate_a_licence(protocol: dict, dataset_manifest: dict) -> dict:
    dataset = protocol["dataset"]
    paths = _partition_paths(dataset)
    corpus = load_jsonl("data/code_corpus_v2.jsonl")
    corpus_hashes: dict[str, set[str]] = {}
    for record in corpus:
        corpus_hashes.setdefault(record.get("source_id"), set()).add(
            record.get("content_sha1"))
    verified = _verified_sources(dataset_manifest)
    problems = []
    counts = {}
    for partition, path in paths.items():
        records = load_jsonl(path)
        counts[partition] = len(records)
        for record in records:
            source = record.get("source")
            if source == FIXTURE_SOURCE:
                if record.get("license") != "CC0-1.0":
                    problems.append(f"{record['id']}: fixture without CC0-1.0")
                continue
            if source not in verified:
                problems.append(f"{record['id']}: unverified source {source!r}")
                continue
            if not record.get("license"):
                problems.append(f"{record['id']}: no licence field")
            if record.get("source_sha") not in corpus_hashes.get(source, set()):
                problems.append(
                    f"{record['id']}: source_sha is not a corpus hash for {source}")
    withheld = set(dataset_manifest["licence_gate"]["withheld_sources"])
    leaked = [p for p in counts if any(
        r.get("source") in withheld for r in load_jsonl(paths[p]))]
    if leaked:
        problems.append(f"withheld sources present in partitions {leaked}")
    return {
        "gate": "A",
        "name": "licence_and_provenance",
        "ok": not problems,
        "verified_sources": sorted(verified),
        "withheld_sources": sorted(withheld),
        "records": counts,
        "problems": problems[:20],
    }


def gate_b_dataset(protocol: dict, dataset_manifest: dict) -> dict:
    from tokenizers import Tokenizer
    from training.sft_dataset import InstructionDataset

    dataset = protocol["dataset"]
    tokenizer_path = dataset_manifest["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    paths = _partition_paths(dataset)
    problems = []
    reports = {}
    oracle_payload = json.loads(Path(dataset["oracles"]).read_text(encoding="utf-8"))
    oracles = {o["example_id"]: o for o in oracle_payload["oracles"]}
    locked_ids = {r["id"] for r in load_jsonl(paths["locked"])}

    for partition, path in paths.items():
        records = load_jsonl(path)
        try:
            encoded = InstructionDataset(path, tokenizer, max_length=512)
        except ValueError as exc:
            problems.append(f"{partition}: encoder refused every example: {exc}")
            continue
        report = encoded.report()
        reports[partition] = {
            "records": report["records_read"],
            "encoded": report["records_encoded"],
            "rejected": report["records_rejected"],
            "truncated": report["truncated"],
            "supervised_tokens": report["supervised_tokens"],
            "rejected_examples": report["rejected_examples"],
        }
        if report["records_rejected"]:
            problems.append(
                f"{partition}: {report['records_rejected']} examples do not fit "
                "the 512-token context")

        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            problems.append(f"{partition}: duplicate ids")

        for record in records:
            task = record["task"]
            if task == "code_generation":
                syntax = score_generation_syntax(record["response"])
                if syntax["status"] != "SYNTAX_VALID":
                    problems.append(
                        f"{record['id']}: reference response not valid code "
                        f"({syntax['status']})")
            elif task == "documentation":
                checked = score_documentation(record["response"])
                if not checked["format_pass"]:
                    problems.append(
                        f"{record['id']}: reference docstring is not docstring-only")
            elif task == "code_explanation":
                checked = score_explanation(record["response"])
                if not checked["format_pass"]:
                    problems.append(
                        f"{record['id']}: reference explanation fails format check")
            elif task == "sql":
                checked = score_sql(record, record["response"])
                if checked["status"] != "PASS":
                    problems.append(
                        f"{record['id']}: reference SQL failed: {checked['status']}")
            elif task == "json_output":
                checked = score_json(record, record["response"])
                if not checked["pass"]:
                    problems.append(f"{record['id']}: reference JSON invalid")
            elif task == "config_output":
                checked = (score_yaml(record, record["response"])
                           if "YAML" in record["instruction"]
                           else score_table(record, record["response"]))
                if not checked["pass"]:
                    problems.append(f"{record['id']}: reference config invalid")

    if len(oracles) < 8:
        problems.append(f"only {len(oracles)} functional oracles; 8 required")
    for example_id, oracle in oracles.items():
        if example_id not in locked_ids:
            problems.append(f"oracle {oracle['function']} is not in locked partition")
        if len(oracle["cases"]) < 3:
            problems.append(f"oracle {oracle['function']} has <3 cases")
        for case in oracle["cases"]:
            if case["expected"].get("status") not in ("ok", "raised"):
                problems.append(
                    f"oracle {oracle['function']} has an unrecorded expectation")

    train_tasks = Counter(r["task"] for r in load_jsonl(paths["train"]))
    locked_tasks = Counter(r["task"] for r in load_jsonl(paths["locked"]))
    if len(load_jsonl(paths["train"])) < 300:
        problems.append("training partition has fewer than 300 records")
    if locked_tasks.get("code_generation", 0) < 20:
        problems.append("locked partition has fewer than 20 generation tasks")
    return {
        "gate": "B",
        "name": "dataset_semantic_quality",
        "ok": not problems,
        "partitions": reports,
        "train_tasks": dict(train_tasks),
        "locked_tasks": dict(locked_tasks),
        "oracles": len(oracles),
        "problems": problems[:20],
    }


def m5_integrity(protocol: dict, train_records=None):
    benchmark = load_m5(protocol["benchmark_integrity"]["m5_benchmark"])
    expected = protocol["benchmark_integrity"]["m5_expected_sha256"]
    actual = m5_manifest_hash(benchmark)
    manifest = json.loads(Path(protocol["benchmark_integrity"]["m5_manifest"])
                          .read_text(encoding="utf-8"))
    unchanged = actual == expected and manifest.get("manifest_hash") == expected
    problems = []
    if not unchanged:
        problems.append(
            f"M5 benchmark hash changed: expected {expected}, got {actual}")
    m5 = {
        "benchmark_examples": len(benchmark),
        "computed_hash": actual,
        "expected_hash": expected,
        "manifest_hash": manifest.get("manifest_hash"),
        "unchanged": unchanged,
        "overlap_policy": ("M5 is not M7's locked evaluation and is not used "
                           "as final evidence; overlaps are disclosed."),
    }
    if train_records is not None:
        train_text = {_norm(r["instruction"]) for r in train_records} | {
            _norm(r["response"]) for r in train_records}
        m5_prompts = {_norm(r.get("prompt", "")) for r in benchmark}
        m5_refs = {_norm(r.get("reference", "")) for r in benchmark
                   if r.get("reference")}
        m5["training_prompt_overlap"] = len(
            sorted((m5_prompts & train_text) - {""}))
        m5["training_reference_overlap"] = len(
            sorted((m5_refs & train_text) - {""}))
    return problems, m5


def gate_c_isolation(protocol: dict, dataset_manifest: dict) -> dict:
    paths = _partition_paths(protocol["dataset"])
    partitions = {p: load_jsonl(path) for p, path in paths.items()}
    problems = []
    pairs = {}
    names = sorted(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = partitions[left], partitions[right]
            a_ids, b_ids = {r["id"] for r in a}, {r["id"] for r in b}
            a_ex = {(r["task"], _norm(r["instruction"]), _norm(r.get("input", "")),
                     _norm(r["response"])) for r in a}
            b_ex = {(r["task"], _norm(r["instruction"]), _norm(r.get("input", "")),
                     _norm(r["response"])) for r in b}
            a_resp = {hashlib.sha256(r["response"].encode()).hexdigest() for r in a}
            b_resp = {hashlib.sha256(r["response"].encode()).hexdigest() for r in b}
            a_src = {r["source"] for r in a if r["source"] != FIXTURE_SOURCE}
            b_src = {r["source"] for r in b if r["source"] != FIXTURE_SOURCE}
            a_file = {r["source_sha"] for r in a
                      if r["source"] != FIXTURE_SOURCE and r.get("source_sha")}
            b_file = {r["source_sha"] for r in b
                      if r["source"] != FIXTURE_SOURCE and r.get("source_sha")}
            source_conflict = (
                ({left, right} & {"train", "locked"})
                and bool(a_src & b_src))
            record = {
                "shared_ids": len(a_ids & b_ids),
                "shared_examples": len(a_ex & b_ex),
                "shared_responses": len(a_resp & b_resp),
                "shared_sources": sorted(a_src & b_src),
                "shared_files": len(a_file & b_file),
                "source_conflict": bool(source_conflict),
                "source_policy": ("val/dev may share a source only when file-level "
                                  "splits are disjoint; train and locked may not."),
            }
            record["ok"] = not (a_ids & b_ids or a_ex & b_ex or a_resp & b_resp
                                or source_conflict or a_file & b_file)
            if not record["ok"]:
                problems.append(f"{left}/{right}: isolation violated: {record}")
            pairs[f"{left}__{right}"] = record

    m5_problems, m5 = m5_integrity(protocol, partitions["train"])
    problems.extend(m5_problems)

    return {
        "gate": "C",
        "name": "split_integrity_and_leakage",
        "ok": not problems,
        "pairs": pairs,
        "m5": m5,
        "problems": problems[:20],
    }


def gate_d_resources(protocol: dict, preflight_path="reports/M7_PREFLIGHT.json") -> dict:
    from training.preflight import run_preflight

    report = run_preflight("configs/train/pretrain_m7.yaml", probe_steps=3)
    path = Path(preflight_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    runtime = report.get("runtime_estimate_seconds")
    required = report.get("memory_required_bytes")
    budget = report.get("memory_budget_bytes")
    total_budget = report.get("memory_total_budget_bytes")
    free = report["system"]["disk"]["free_bytes"]
    problems = []
    warnings = []
    if report["verdict"] not in ("FEASIBLE", "FEASIBLE_SMALLER"):
        problems.append(f"preflight verdict: {report['verdict']}")
    if report.get("hard_gate_failures"):
        problems.append(f"hard gate failures: {report['hard_gate_failures']}")
    if runtime is None or runtime > 90 * 60:
        problems.append(f"pretraining projected runtime {runtime}s exceeds 90 min")
    if required is None or total_budget is None or required > total_budget:
        problems.append(
            f"memory required {required} exceeds stable machine budget {total_budget}")
    elif budget is not None and required > budget:
        warnings.append(
            f"current available RAM is tight ({required} required vs {budget} "
            "budget); preflight says the run fits the machine under pressure")
    if free < 1 << 30:
        problems.append(f"less than 1 GiB free disk: {free}")
    return {
        "gate": "D",
        "name": "resource_feasibility",
        "ok": not problems,
        "verdict": report["verdict"],
        "under_memory_pressure": bool(budget is not None and required > budget),
        "warnings": warnings,
        "preflight_report": str(path),
        "preflight_sha256": sha256_file(path),
        "params": report["model"]["params"],
        "memory_required_bytes": required,
        "memory_budget_bytes": budget,
        "memory_total_budget_bytes": total_budget,
        "peak_rss_bytes": (report.get("probe") or {}).get("peak_rss_bytes"),
        "tokens_per_second": (report.get("probe") or {}).get("tokens_per_second"),
        "runtime_estimate_seconds": runtime,
        "runtime_estimate_minutes": round(runtime / 60, 1) if runtime else None,
        "sft_runtime_estimate_minutes": 12,
        "sft_runtime_basis": ("M4 measured 216 s for 120 steps at 2.85M; scaled "
                              "by the 9.36M/2.85M parameter ratio. ESTIMATE."),
        "problems": problems,
    }


def gate_e_authorization(protocol: dict, pre_results, resume=False) -> dict:
    problems = []
    for gate in pre_results:
        if not gate["ok"]:
            problems.append(f"gate {gate['gate']} failed")
    for run in ("runs/m7_pretrain/checkpoint_last.pt",
                "runs/m7_sft/checkpoint_last.pt"):
        if Path(run).exists() and not resume:
            problems.append(f"existing run found: {run}; pass --resume deliberately")
    for name, expected in protocol["historical_checkpoint_sha256"].items():
        path = protocol["checkpoints"][name]
        if not Path(path).exists():
            problems.append(f"historical checkpoint missing: {path}")
        elif sha256_file(path).upper() != expected.upper():
            problems.append(f"historical checkpoint changed: {path}")
    free = __import__("shutil").disk_usage(".").free
    if free < 2 << 30:
        problems.append(f"less than 2 GiB free disk: {free}")
    inputs = {
        "protocol_sha256": sha256_file("configs/eval/m7_protocol.yaml"),
        "dataset_manifest_sha256": sha256_file(protocol["dataset"]["manifest"]),
        "pretrain_config_sha256": sha256_file("configs/train/pretrain_m7.yaml"),
        "sft_config_sha256": sha256_file("configs/train/sft_m7.yaml"),
        "gates": {g["gate"]: g["ok"] for g in pre_results},
    }
    token = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    authorization = {
        "schema": "quantsmind.m7_authorization/1",
        "generated_at": utc_now_iso(),
        "authorized": not problems,
        "inputs": inputs,
        "token": token,
        "problems": problems,
        "note": ("This is a content-addressed authorization record, not a "
                 "cryptographic signature. It binds the dataset, protocol, "
                 "configs, and gate results."),
    }
    Path("reports/M7_AUTHORIZATION.json").write_text(
        json.dumps(authorization, indent=2), encoding="utf-8")
    return {
        "gate": "E",
        "name": "training_authorization",
        "ok": not problems,
        "authorization": authorization,
        "problems": problems,
    }


def _run_manifest(name):
    path = Path(f"runs/{name}/run_manifest.json")
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _reload_checkpoint(path) -> dict:
    script = (
        "import json\n"
        "from training.generate import load_checkpoint_for_generation\n"
        f"m, i = load_checkpoint_for_generation({str(path)!r})\n"
        "print(json.dumps({'parameters_loaded': i['parameters_loaded'],"
        " 'step': i['checkpoint_step'], 'parameters': i['parameters']}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=120, cwd=Path.cwd())
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-400:]}
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": proc.stdout[-400:]}
    payload["ok"] = True
    return payload


def gate_f_training(protocol: dict) -> dict:
    problems, runs = [], {}
    for name in ("m7_pretrain", "m7_sft"):
        manifest, path = _run_manifest(name)
        if manifest is None:
            problems.append(f"missing run manifest: {path}")
            continue
        training = manifest.get("training") or {}
        metrics = training.get("metrics") or {}
        history = training.get("history") or []
        initial = history[0].get("train_loss") if history else None
        final = metrics.get("final_train_loss")
        runs[name] = {
            "manifest": str(path),
            "status": manifest.get("status"),
            "success": manifest.get("success"),
            "steps_completed": training.get("steps_completed"),
            "planned_steps": training.get("planned_steps"),
            "final_train_loss": final,
            "best_validation_loss": metrics.get("best_validation_loss"),
            "initial_train_loss": initial,
            "checkpoint": manifest.get("checkpoint"),
            "dataset": manifest.get("dataset"),
        }
        if manifest.get("status") != "complete" or manifest.get("success") is not True:
            problems.append(f"{name}: status={manifest.get('status')} "
                            f"success={manifest.get('success')}")
        if training.get("steps_completed") != training.get("planned_steps"):
            problems.append(f"{name}: steps did not complete")
        if final is None or not isinstance(final, float):
            problems.append(f"{name}: final train loss missing")
        elif final >= UNIFORM_LOSS:
            problems.append(f"{name}: final train loss {final} not below uniform")
        if name == "m7_pretrain" and metrics.get("best_validation_loss") is None:
            problems.append("m7_pretrain: no validation loss")
        if name == "m7_sft" and initial is not None and final is not None \
                and final >= initial:
            problems.append(f"m7_sft: loss did not fall ({initial} -> {final})")
        checkpoint = manifest.get("checkpoint") or {}
        ckpt_path = checkpoint.get("path")
        if not ckpt_path or not Path(ckpt_path).exists():
            problems.append(f"{name}: checkpoint missing: {ckpt_path}")
        else:
            reloaded = _reload_checkpoint(ckpt_path)
            runs[name]["fresh_process_reload"] = reloaded
            if not reloaded.get("ok"):
                problems.append(f"{name}: checkpoint reload failed")
        dataset = manifest.get("dataset") or {}
        declared = dataset.get("manifest_path")
        if not declared or not Path(declared).exists():
            problems.append(f"{name}: dataset manifest missing: {declared}")
        elif sha256_file(declared) != dataset.get("manifest_sha256"):
            problems.append(f"{name}: declared dataset manifest hash mismatch")
        expected_manifest = ("data/m7/pretrain.manifest.json"
                             if name == "m7_pretrain"
                             else protocol["dataset"]["manifest"])
        if declared != expected_manifest:
            problems.append(
                f"{name}: trained on {declared}, expected {expected_manifest}")
        for artifact, info in (dataset.get("artifacts") or {}).items():
            if not Path(artifact).exists():
                problems.append(f"{name}: missing artifact {artifact}")
            elif sha256_file(artifact) != info.get("sha256"):
                problems.append(f"{name}: artifact hash mismatch {artifact}")
    for name, expected in protocol["historical_checkpoint_sha256"].items():
        path = protocol["checkpoints"][name]
        if not Path(path).exists() or sha256_file(path).upper() != expected.upper():
            problems.append(f"historical checkpoint changed or missing: {path}")
    return {
        "gate": "F",
        "name": "training_integrity",
        "ok": not problems,
        "runs": runs,
        "problems": problems[:20],
    }


def _model_tasks(summary, model_name, partition="locked"):
    for model in summary.get("models", []):
        if model["model"] == model_name:
            return model.get("partitions", {}).get(partition, {}).get("tasks", {})
    return {}


def common_validation_losses(protocol: dict) -> dict:
    """Score every checkpoint on the same M7 validation windows.

    M3's recorded validation loss was measured on a different aiohttp subset,
    so comparing the two numbers directly would be invalid. These losses all use
    data/m7/pretrain.val.tokens.bin, 8 fixed windows of 512 tokens, seed 1234.
    """
    from training.data import TokenDataset
    from training.generate import load_checkpoint_for_generation
    from training.validation import evaluate

    val_path = "data/m7/pretrain.val.tokens.bin"
    losses = {}
    for name in ("untrained", "m3_base", "m4_tuned", "m7_candidate"):
        checkpoint = protocol["checkpoints"].get(name)
        if not checkpoint or not Path(checkpoint).exists():
            losses[name] = {"status": "MISSING", "loss": None}
            continue
        try:
            model, info = load_checkpoint_for_generation(checkpoint)
            dataset = TokenDataset(val_path, seed=1234)
            result = evaluate(model, dataset, seq_len=512, batches=8, seed=1234,
                              device="cpu", batch_size=1)
            losses[name] = {
                "status": "ok" if not result.errors else "ERROR",
                "loss": result.loss,
                "errors": result.errors,
                "parameters": info["parameters"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            losses[name] = {"status": "ERROR", "loss": None,
                            "error": f"{type(exc).__name__}: {exc}"}
    return {
        "stream": val_path,
        "seq_len": 512,
        "batches": 8,
        "seed": 1234,
        "losses": losses,
    }


def gate_g_evidence(protocol: dict) -> dict:
    path = Path("runs/m7_eval/summary.json")
    if not path.exists():
        return {"gate": "G", "name": "capability_evidence", "ok": False,
                "status": "NOT_RUN", "problems": ["evaluation summary missing"],
                "evidence_level": 0}
    summary = json.loads(path.read_text(encoding="utf-8"))
    candidate = _model_tasks(summary, "m7_candidate")
    baselines = {name: _model_tasks(summary, name)
                 for name in ("untrained", "m3_base", "m4_tuned")}
    generation = candidate.get("code_generation", {})
    syntax = generation.get("syntax_rate")
    functional = generation.get("functional_rate")
    attempted = generation.get("attempted", 0)
    base_syntax = max((b.get("code_generation", {}).get("syntax_rate") or 0.0)
                      for b in baselines.values())
    base_functional = max((b.get("code_generation", {}).get("functional_rate") or 0.0)
                          for b in baselines.values())

    validation = common_validation_losses(protocol)
    losses = validation["losses"]
    val_loss = (losses.get("m7_candidate") or {}).get("loss")
    control_losses = {
        name: (losses.get(name) or {}).get("loss")
        for name in ("untrained", "m3_base", "m4_tuned")
    }
    usable_controls = {name: value for name, value in control_losses.items()
                       if isinstance(value, float)}
    best_control = min(usable_controls.values()) if usable_controls else None
    level = 0
    rationale = []
    if val_loss is not None and best_control is not None:
        rationale.append(
            "common-validation loss "
            f"{val_loss:.4f} vs controls "
            + ", ".join(
                f"{name}={value:.4f}" if isinstance(value, float)
                else f"{name}=MISSING"
                for name, value in control_losses.items()))
        if val_loss < best_control and val_loss < UNIFORM_LOSS:
            level = 1
        else:
            rationale.append("did not beat every control on the same windows")

    thresholds = protocol["thresholds"]
    if level >= 1 and syntax is not None \
            and syntax >= thresholds["syntax_rate_min"] \
            and attempted >= thresholds["min_locked_generation"] \
            and syntax >= base_syntax + thresholds["min_absolute_gain"]:
        level = 2
        rationale.append(f"locked syntax {syntax:.3f} vs best baseline {base_syntax:.3f}")
    if level >= 2 and functional is not None \
            and generation.get("functional_attempted", 0) >= thresholds["min_functional_tasks"] \
            and functional >= thresholds["functional_pass_min"] \
            and functional >= base_functional + thresholds["min_absolute_gain"]:
        level = 3
        rationale.append(
            f"locked functional {functional:.3f} vs best baseline {base_functional:.3f}")

    categories = {}
    for task, entry in candidate.items():
        if task == "code_generation":
            categories[task] = {
                "rate": entry.get("functional_rate"),
                "attempted": entry.get("functional_attempted", 0)}
        elif task == "sql":
            categories[task] = {"rate": entry.get("sql_rate"),
                                "attempted": entry.get("sql_attempted", 0)}
        elif task in ("json_output", "config_output"):
            categories[task] = {"rate": entry.get("structured_rate"),
                                "attempted": entry.get("structured_attempted", 0)}
        elif task in ("format_following", "constrained_output"):
            categories[task] = {"rate": entry.get("instruction_rate"),
                                "attempted": entry.get("instruction_attempted", 0)}
    successful = [name for name, info in categories.items()
                  if info["rate"] is not None and info["rate"] >= 0.5
                  and info["attempted"] >= 3]
    if level >= 3 and len(successful) >= thresholds["category_count_for_level4"]:
        level = 4
        rationale.append(f"multiple categories succeeded: {successful}")
    if level >= 4 and thresholds["independent_review_required"]:
        rationale.append("no independent human review artifact; level 5 not claimed")
    if level >= 4:
        outcome = "MULTI_CATEGORY_CAPABILITY_DEMONSTRATED"
    elif level == 3:
        outcome = "FUNCTIONAL_CAPABILITY_DEMONSTRATED"
    elif level == 2:
        outcome = "SYNTAX_CAPABILITY_DEMONSTRATED"
    elif level == 1:
        outcome = "LEARNING_DEMONSTRATED"
    elif val_loss is None:
        outcome = "INSUFFICIENT_EVIDENCE"
    else:
        outcome = "TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED"
    return {
        "gate": "G",
        "name": "capability_evidence",
        "ok": level >= 2,
        "status": outcome,
        "evidence_level": level,
        "rationale": rationale,
        "locked_generation": {
            "attempted": attempted,
            "syntax_rate": syntax,
            "functional_rate": functional,
            "functional_attempted": generation.get("functional_attempted"),
            "functional_passed": generation.get("functional_passed"),
            "best_baseline_syntax": base_syntax,
            "best_baseline_functional": base_functional,
        },
        "common_validation": validation,
        "common_validation_best_control": best_control,
        "categories": categories,
        "successful_categories": successful,
        "problems": [] if level >= 2 else ["capability level 2 not reached"],
    }


def gate_h_independence(protocol: dict, gate_c: dict) -> dict:
    problems = []
    if not gate_c.get("ok"):
        problems.append("split/leakage gate C failed")
    if not gate_c.get("m5", {}).get("unchanged"):
        problems.append("M5 benchmark is not unchanged")
    summary_path = Path("runs/m7_eval/summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol_sha256") != sha256_file("configs/eval/m7_protocol.yaml"):
            problems.append("evaluation used a different protocol hash")
        for model in summary.get("models", []):
            for record in load_jsonl(Path("runs/m7_eval") / model["model"] /
                                     "records.jsonl")[:5]:
                if not record.get("checkpoint_sha256"):
                    problems.append(f"{model['model']}: records lack checkpoint hash")
                    break
    else:
        problems.append("evaluation summary missing")
    return {
        "gate": "H",
        "name": "independent_evaluation",
        "ok": not problems,
        "locked_source": protocol["dataset"]["locked_source"],
        "human_review": False,
        "level_5_available": False,
        "problems": problems,
    }


def build_report(phase="pre", resume=False) -> dict:
    protocol = load_protocol()
    dataset_manifest = json.loads(
        Path(protocol["dataset"]["manifest"]).read_text(encoding="utf-8"))
    if phase == "pre":
        gates = [
            gate_a_licence(protocol, dataset_manifest),
            gate_b_dataset(protocol, dataset_manifest),
            gate_c_isolation(protocol, dataset_manifest),
            gate_d_resources(protocol),
        ]
        gates.append(gate_e_authorization(protocol, gates, resume=resume))
    else:
        gates = [
            gate_f_training(protocol),
            gate_g_evidence(protocol),
            gate_h_independence(protocol, gate_c_isolation(protocol, dataset_manifest)),
        ]
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now_iso(),
        "phase": phase,
        "gates": gates,
        "ok": all(g["ok"] for g in gates),
    }
    path = Path("reports/M7_GATES.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    if phase == "post":
        gate_g = next((g for g in gates if g["gate"] == "G"), {})
        outcome = {
            "schema": "quantsmind.m7_outcome/1",
            "generated_at": report["generated_at"],
            "outcome": gate_g.get("status", "INSUFFICIENT_EVIDENCE"),
            "evidence_level": gate_g.get("evidence_level", 0),
            "gates": {g["gate"]: g["ok"] for g in gates},
            "locked_generation": gate_g.get("locked_generation"),
            "common_validation": gate_g.get("common_validation"),
            "limitations": [
                "baselines were evaluated on locked code_generation only",
                "6/14 locked completion prompts exceed the 512-token context",
                "candidate secondary tasks used max_new_tokens=128 (protocol 192)",
                "no independent human review; evidence level 5 unavailable",
                "single seed, CPU-only run",
            ],
        }
        Path("reports/M7_OUTCOME.json").write_text(
            json.dumps(outcome, indent=2), encoding="utf-8")
        report["outcome_path"] = "reports/M7_OUTCOME.json"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run M7 gates")
    parser.add_argument("--phase", choices=("pre", "post"), default="pre")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.phase, resume=args.resume)
    for gate in report["gates"]:
        print(f"[{'PASS' if gate['ok'] else 'FAIL'}] {gate['gate']} "
              f"{gate['name']}")
        for problem in gate.get("problems", [])[:5]:
            print(f"      {problem}")
    print(f"report: {report['report_path']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
