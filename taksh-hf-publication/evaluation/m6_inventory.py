"""M6 candidate inventory: what models actually exist on this machine.

Discovers candidates by *looking at the filesystem*, never by trusting a
declared list. Anything the repository claims but that cannot be found on disk
is reported as ``NOT_PRESENT`` rather than assumed to exist.

States
------
``AVAILABLE_LOCAL``   weights found and complete enough to inspect
``EVALUATED``         real inference ran and outputs were scored
``NOT_EVALUATED``     present and eligible, but not yet run
``ELIGIBLE``          licensing and provenance cleared
``BLOCKED``           a specific, stated reason prevents evaluation
``NOT_PRESENT``       no local artifact and download not authorised

A candidate may only be ``EVALUATED`` when :mod:`evaluation.m6_evaluate` has
persisted real outputs and scored them. Discovery alone never sets that state.

    python -m evaluation.m6_inventory
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m6_resources import GIB, MIB, machine_facts, plan_for

# A snapshot is only usable when the weight files are complete. Hugging Face
# leaves partial downloads as "<blob>.incomplete", and a sharded model can have
# an index that references shards which never arrived. Both are traps.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def hf_cache_root() -> Path:
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env)
        if value:
            return Path(value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class SnapshotFile:
    name: str
    bytes: int
    complete: bool
    kind: str


@dataclass
class Candidate:
    """One assessed model. Every field is either measured or explicitly unknown."""

    candidate_id: str
    publisher: str | None = None
    source_repository: str | None = None
    revision: str | None = None
    architecture: str | None = None
    parameters: int | None = None
    weight_format: str | None = None
    quantization: str | None = None
    tokenizer_requirement: str | None = None
    prompt_template: str | None = None
    license_id: str | None = None
    license_source: str | None = None
    license_verified: bool = False
    tokenizer_license_id: str | None = None
    acquisition_status: str = "UNKNOWN"
    local_available: bool = False
    local_path: str | None = None
    artifact_bytes: int = 0
    runtime_compatible: bool = False
    runtime_reason: str = ""
    est_ram_bytes: int | None = None
    est_disk_gib: float | None = None
    measured_requirements: dict = field(default_factory=dict)
    intended_use_restrictions: str | None = None
    task_head: str = "unknown"
    evaluation_status: str = "NOT_EVALUATED"
    blocking_reason: str | None = None
    remaining_uncertainties: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["artifact_mib"] = round(self.artifact_bytes / MIB, 1)
        data["est_ram_gib"] = (round(self.est_ram_bytes / GIB, 3)
                               if self.est_ram_bytes else None)
        return data


# --------------------------------------------------------------------------- #
# Filesystem discovery
# --------------------------------------------------------------------------- #
def _classify(name: str, size: int, snapshot_root: Path) -> tuple[bool, str]:
    """Is this snapshot complete enough to load? Returns ``(ok, kind)``."""
    if name.endswith(".incomplete"):
        return False, "partial_download"
    if name == "model.safetensors.index.json":
        return True, "shard_index"
    if name.endswith(WEIGHT_SUFFIXES):
        return True, "weights"
    if name == "config.json":
        return True, "config"
    if name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                "vocab.json", "merges.txt", "tokenizer.model"):
        return True, "tokenizer"
    return True, "metadata"


def inspect_snapshot(snapshot: Path) -> dict:
    """Inspect one HF cache snapshot without importing any model code."""
    files, total, kinds = [], 0, set()
    incomplete = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        ok, kind = _classify(path.name, size, snapshot)
        files.append(SnapshotFile(path.name, size, ok, kind))
        kinds.add(kind)
        total += size
        if not ok:
            incomplete.append(path.name)

    index = snapshot / "model.safetensors.index.json"
    missing_shards: list[str] = []
    declared_size = None
    if index.exists():
        try:
            meta = json.loads(index.read_text(encoding="utf-8"))
            shards = sorted(set(meta.get("weight_map", {}).values()))
            missing_shards = [s for s in shards
                              if not (snapshot / s).exists()]
            declared_size = (meta.get("metadata") or {}).get("total_size")
        except (OSError, ValueError):
            missing_shards = ["<index unreadable>"]

    weights = [f for f in files if f.kind == "weights"]
    has_tokenizer = "tokenizer" in kinds
    has_config = "config" in kinds

    complete = bool(weights) and has_config and not incomplete \
        and not missing_shards
    return {
        "snapshot": str(snapshot),
        "files": [asdict(f) for f in files],
        "artifact_bytes": sum(f.bytes for f in weights),
        "on_disk_bytes": total,
        "has_config": has_config,
        "has_tokenizer": has_tokenizer,
        "has_weights": bool(weights),
        "incomplete_files": incomplete,
        "missing_shards": missing_shards,
        "declared_total_size": declared_size,
        "complete": complete,
        "weight_format": next((f.name.rsplit(".", 1)[-1] for f in weights), None),
    }


def discover_hf_candidates(cache_root=None) -> list[dict]:
    """Every model repository present in the Hugging Face cache."""
    root = Path(cache_root) if cache_root else hf_cache_root()
    found = []
    if not root.exists():
        return found
    for repo in sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name.startswith("models--")):
        model_id = repo.name[len("models--"):].replace("--", "/")
        snapshots_dir = repo / "snapshots"
        snapshots = ([s for s in sorted(snapshots_dir.iterdir())
                      if s.is_dir()] if snapshots_dir.exists() else [])
        # ``refs/main`` names the revision the download actually recorded, so it
        # wins over directory ordering. Falling back to name order alone picked
        # a stale snapshot because "old" sorts after "new".
        ref_file = repo / "refs" / "main"
        preferred = None
        if ref_file.exists():
            try:
                preferred = (snapshots_dir / ref_file.read_text(
                    encoding="utf-8").strip())
            except OSError:
                preferred = None
        chosen = preferred if preferred and preferred.is_dir() else (
            snapshots[-1] if snapshots else None)
        if chosen is None:
            found.append({
                "model_id": model_id, "local_path": str(repo),
                "revision": None, "complete": False,
                "reason": "repository directory exists but contains no snapshot",
                "artifact_bytes": 0,
            })
            continue
        revision = chosen.name
        detail = inspect_snapshot(chosen)
        detail.update({"model_id": model_id, "revision": revision,
                       "local_path": str(chosen)})
        if ref_file.exists():
            try:
                detail["revision_from_ref"] = ref_file.read_text(
                    encoding="utf-8").strip()
            except OSError:
                pass
        found.append(detail)
    return found


# --------------------------------------------------------------------------- #
# Registry of externally interesting candidates we want to know about even
# when they are absent from disk.
# --------------------------------------------------------------------------- #
WATCHLIST = [
    "microsoft/Phi-3-mini-4k-instruct",
    "google-bert/bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

# The cache directory name encodes whatever repository id was used at download
# time, and older releases used the legacy un-namespaced alias. Without this
# mapping a perfectly good local snapshot is reported as NOT_PRESENT.
ALIASES = {
    "bert-base-uncased": "google-bert/bert-base-uncased",
}

SOVEREIGN_CHECKPOINTS = [
    ("m3-base", "runs/m3_base/checkpoint_best.pt", 2_853_280),
    ("m4-tuned", "runs/m4_sft/checkpoint_best.pt", 2_853_280),
]


def _runtime_available() -> tuple[bool, str]:
    try:
        import transformers
        return True, f"transformers {transformers.__version__}"
    except ImportError as exc:
        return False, f"transformers is not installed ({exc})"


def build_candidate(model_id, discovery, *, runtime_ok, runtime_reason,
                    plan=None) -> Candidate:
    """Turn a filesystem discovery (or its absence) into an assessed candidate."""
    from evaluation.m6_licensing import lookup

    reg = lookup(model_id) or {}
    present = bool(discovery) and bool(discovery.get("complete"))
    incomplete_reason = None
    if discovery and not discovery.get("complete"):
        bits = []
        if discovery.get("incomplete_files"):
            bits.append(f"partial downloads: {discovery['incomplete_files']}")
        if discovery.get("missing_shards"):
            bits.append(f"missing shards: {discovery['missing_shards']}")
        if not discovery.get("has_weights"):
            bits.append("no weight file present")
        if not discovery.get("has_config"):
            bits.append("no config.json present")
        incomplete_reason = "; ".join(bits) or discovery.get("reason") \
            or "snapshot incomplete"

    cand = Candidate(
        candidate_id=model_id,
        publisher=reg.get("publisher"),
        source_repository=f"https://huggingface.co/{model_id}",
        revision=(discovery or {}).get("revision"),
        architecture=reg.get("architecture"),
        parameters=reg.get("parameters"),
        weight_format=(discovery or {}).get("weight_format"),
        quantization=reg.get("quantization"),
        tokenizer_requirement=reg.get("tokenizer_requirement"),
        prompt_template=reg.get("prompt_template"),
        license_id=reg.get("license_id"),
        license_source=reg.get("license_source"),
        license_verified=bool(reg.get("license_verified")),
        tokenizer_license_id=reg.get("tokenizer_license_id"),
        intended_use_restrictions=reg.get("intended_use_restrictions"),
        task_head=reg.get("task_head", "unknown"),
        acquisition_status=(
            "NOT_PRESENT" if discovery is None
            else "PARTIAL" if discovery and not discovery.get("complete")
            else "PRESENT"),
        local_available=present,
        local_path=(discovery or {}).get("local_path"),
        artifact_bytes=(discovery or {}).get("artifact_bytes", 0),
        runtime_compatible=runtime_ok,
        runtime_reason=runtime_reason,
    )
    if plan is not None:
        cand.est_ram_bytes = plan.required_ram_bytes
        cand.est_disk_gib = round(plan.required_disk_bytes / GIB, 3)

    # Status is decided by the first rule that applies, in severity order.
    if not reg:
        cand.evaluation_status = "BLOCKED"
        cand.blocking_reason = (
            "no verified licence or architecture record in "
            "evaluation/m6_licensing.py; pending human review")
    elif not cand.license_verified:
        cand.evaluation_status = "BLOCKED"
        cand.blocking_reason = "licence terms unresolved; pending human review"
    elif discovery is None:
        cand.evaluation_status = "NOT_PRESENT"
        cand.blocking_reason = (
            "no local artifact; downloading weights requires explicit "
            "authorisation and is not performed by M6")
    elif not present:
        cand.evaluation_status = "BLOCKED"
        cand.blocking_reason = f"incomplete snapshot: {incomplete_reason}"
    elif not runtime_ok:
        cand.evaluation_status = "BLOCKED"
        cand.blocking_reason = f"runtime incompatible: {runtime_reason}"
    elif plan is not None and not plan.ok:
        ok, reason = plan.decision()
        cand.evaluation_status = "BLOCKED"
        cand.blocking_reason = f"resource preflight refused: {reason}"
    else:
        cand.evaluation_status = "NOT_EVALUATED"
        cand.blocking_reason = None

    if cand.task_head in ("embedding", "masked_lm"):
        cand.remaining_uncertainties.append(
            f"task head is {cand.task_head}, which cannot answer a free-form "
            f"coding instruction; not a practical coding candidate")
    if cand.task_head == "causal_lm" and cand.acquisition_status == "NOT_PRESENT":
        cand.remaining_uncertainties.append(
            "capability on the M5 benchmark is entirely unmeasured")
    return cand


def build_sovereign_candidates() -> list[Candidate]:
    """The project's own checkpoints, which are not external candidates."""
    out = []
    for label, path, params in SOVEREIGN_CHECKPOINTS:
        p = Path(path)
        exists = p.exists()
        cand = Candidate(
            candidate_id=f"quantsmind/{label}",
            publisher="QuantsMind-LLM (this repository)",
            source_repository="local repository artifact",
            architecture="GPT-style decoder (project implementation)",
            parameters=params if exists else None,
            weight_format="pytorch .pt" if exists else None,
            license_id="project-owned",
            license_source="data/licence_verification.json (corpus provenance)",
            license_verified=True,
            acquisition_status="PRESENT" if exists else "NOT_PRESENT",
            local_available=exists,
            local_path=path if exists else None,
            artifact_bytes=p.stat().st_size if exists else 0,
            runtime_compatible=True,
            runtime_reason="project checkpoint loader (training.checkpoint)",
            task_head="causal_lm",
            evaluation_status="NOT_EVALUATED",
        )
        if not exists:
            cand.blocking_reason = f"checkpoint not found at {path}"
        out.append(cand)
    return out


def build_inventory(cache_root=None, *, allow_download=False) -> dict:
    """The full M6 candidate inventory.

    Discovery is read-only. ``allow_download`` defaults to False and no code
    path in this module will fetch a weight file under any setting; the flag
    exists purely so a caller can record that authorisation was considered.
    """
    from evaluation.m6_licensing import REGISTRY

    discoveries = {d["model_id"]: d for d in discover_hf_candidates(cache_root)}
    # Fold legacy cache aliases onto their canonical repository ids.
    for alias, canonical in ALIASES.items():
        if alias in discoveries and canonical not in discoveries:
            record = discoveries[alias]
            record["cached_as"] = alias
            record["model_id"] = canonical
            discoveries[canonical] = record
    runtime_ok, runtime_reason = _runtime_available()

    candidates = []
    for model_id in WATCHLIST:
        reg = REGISTRY.get(model_id, {})
        plan = None
        if reg.get("parameters"):
            plan = plan_for(
                model_id, reg["parameters"], dtype=reg.get("dtype", "bfloat16"),
                context=reg.get("context", 512), layers=reg.get("layers"),
                kv_heads=reg.get("kv_heads"), head_dim=reg.get("head_dim"))
        candidates.append(build_candidate(
            model_id, discoveries.get(model_id), runtime_ok=runtime_ok,
            runtime_reason=runtime_reason, plan=plan))

    return {
        "schema": "quantsmind.m6_inventory/1",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "machine": machine_facts(),
        "download_authorised": allow_download,
        "discovery_method": (
            "read-only scan of the Hugging Face cache; no network access, no "
            "model code executed"),
        "candidates": [c.to_dict() for c in candidates],
        "sovereign_checkpoints": [c.to_dict()
                                  for c in build_sovereign_candidates()],
        "cache_root": str(Path(cache_root) if cache_root else hf_cache_root()),
        "summary": {
            "total": len(candidates),
            "evaluated": sum(1 for c in candidates
                             if c.evaluation_status == "EVALUATED"),
            "blocked": sum(1 for c in candidates
                           if c.evaluation_status == "BLOCKED"),
            "not_present": sum(1 for c in candidates
                               if c.evaluation_status == "NOT_PRESENT"),
            "not_evaluated": sum(1 for c in candidates
                                 if c.evaluation_status == "NOT_EVALUATED"),
            "eligible_causal_lm": [
                c.candidate_id for c in candidates
                if c.task_head == "causal_lm"
                and c.evaluation_status in ("NOT_EVALUATED", "EVALUATED")],
        },
    }


def format_inventory(report) -> str:
    lines = ["M6 candidate inventory"]
    machine = report["machine"]
    lines.append(f"  host: {machine['total_ram_gib']:.2f} GiB RAM total, "
                 f"{machine['available_ram_gib']:.2f} GiB available, "
                 f"cuda={machine['torch'].get('cuda_available')}")
    lines.append(f"  cache: {report['cache_root']}")
    lines.append("")
    for cand in report["candidates"]:
        lines.append(f"  [{cand['evaluation_status']}] {cand['candidate_id']}")
        lines.append(f"      acquisition : {cand['acquisition_status']}"
                     f"   head: {cand['task_head']}")
        lines.append(f"      license     : {cand['license_id']} "
                     f"(verified={cand['license_verified']})")
        if cand["est_ram_gib"]:
            lines.append(f"      est RAM     : {cand['est_ram_gib']:.2f} GiB "
                         f"(ESTIMATE)   artifact: "
                         f"{cand['artifact_mib']:.0f} MiB")
        if cand["blocking_reason"]:
            lines.append(f"      blocked by  : {cand['blocking_reason']}")
        for note in cand["remaining_uncertainties"]:
            lines.append(f"      note        : {note}")
        lines.append("")
    for cand in report["sovereign_checkpoints"]:
        lines.append(f"  [{cand['evaluation_status']}] {cand['candidate_id']} "
                     f"(sovereign, {cand['parameters']} parameters)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M6 candidate inventory: what models exist locally")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true",
                        help="print JSON instead of a summary")
    args = parser.parse_args(argv)

    report = build_inventory(args.cache_root)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json
          else format_inventory(report))
    if args.out:
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())