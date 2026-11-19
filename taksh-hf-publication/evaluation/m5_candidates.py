"""Candidate readiness for external open-weight models.

Preflight answers one question: *could this model be evaluated fairly here?* It
deliberately does not compare models. ``transformers`` already refuses to import
anything this repo is not licensed to use, so this module only reports
readiness -- what is missing, and whether the blocker is fixable.

    python -m evaluation.m5_candidates
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Parameters are the model's nominal count. RAM needs are fp16 weights plus a
# margin for the KV cache and the host interpreter.
CANDIDATES = [
    {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "qwen",
        "parameters": 1_543_714_304,
        "license_id": "Apache-2.0",
        "license_verified": True,
        "dtype": "bfloat16",
        "weight_bytes": 3_087_428_608,
        "context_window": 32768,
        "why_candidate": (
            "Smallest well-regarded code-specialised open-weight model, so it is "
            "the cheapest external reference point that a 2.85M-parameter model "
            "could plausibly be compared against."
        ),
    },
]

RAM_REQUIRED_BYTES = 4 * 1024 ** 3
DOWNLOAD_NOT_PERMITTED = (
    "Fetching weights is outside the licence boundary this run operates under; "
    "record as blocked rather than fetch."
)


@dataclass
class Finding:
    """One readiness check. ``ok`` means no further work needed."""

    check: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class Readiness:
    model_id: str
    family: str
    parameters: int
    license_id: str | None
    license_verified: bool
    ready: bool
    findings: list = field(default_factory=list)

    @property
    def blockers(self) -> list:
        return [f for f in self.findings if not f.ok and f.blocking]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["blockers"] = [f.check for f in self.blockers]
        return data


def read_only_finds_local_weights(model_id, extra_paths=()) -> bool:
    """Search the HF cache and any supplied paths. Opens nothing."""
    import glob

    needles = [model_id, model_id.replace("/", "--"),
               model_id.split("/")[-1].replace("-", "--"),
               model_id.split("/")[-1]]
    roots = [Path.home() / ".cache" / "huggingface", Path("models")]
    for base in list(extra_paths) + roots:
        for needle in needles:
            for hit in glob.glob(str(Path(base) / "hub" / f"{needle}*")):
                if Path(hit).exists():
                    return True
    return False


def assess(candidate, extra_paths=()) -> Readiness:
    findings = []

    findings.append(Finding(
        "license",
        ok=bool(candidate["license_verified"] and candidate.get("license_id")),
        detail=(f"{candidate.get('license_id')} (verified in the repository "
                f"licence register)" if candidate["license_verified"]
                else "no verified licence record; not evaluable"),
        blocking=True))

    try:
        import transformers  # noqa: F401
        version = transformers.__version__
        findings.append(Finding(
            "runtime", ok=True,
            detail=f"transformers {version} importable on this interpreter",
            blocking=True))
    except ImportError as exc:
        version = None
        findings.append(Finding(
            "runtime", ok=False, detail=f"transformers unavailable ({exc})",
            blocking=True))

    local = read_only_finds_local_weights(candidate["model_id"], extra_paths)
    findings.append(Finding(
        "weights_local", ok=local,
        detail=("local weights found" if local
                else f"no local weights for {candidate['model_id']}"),
        blocking=True))
    if not local:
        findings.append(Finding(
            "download_policy", ok=False, detail=DOWNLOAD_NOT_PERMITTED,
            blocking=True))

    needed = candidate["weight_bytes"]
    findings.append(Finding(
        "ram", ok=False,
        detail=(f"needs ~{needed / 1024 ** 3:.1f} GiB for {candidate['dtype']} "
                f"weights plus KV cache; decision to attempt a download is "
                f"required before memory can be judged"),
        blocking=True))

    return Readiness(
        model_id=candidate["model_id"], family=candidate["family"],
        parameters=candidate["parameters"],
        license_id=candidate.get("license_id"),
        license_verified=bool(candidate["license_verified"]),
        ready=all(f.ok for f in findings), findings=findings)


def assess_all(extra_paths=()) -> list:
    return [assess(c, extra_paths) for c in CANDIDATES]


def main(argv=None) -> int:
    import platform

    out = Path("reports/M5_CANDIDATE_READINESS.json")
    results = assess_all()
    payload = {
        "schema": "quantsmind.m5_candidate_readiness/1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ram_required_bytes": RAM_REQUIRED_BYTES,
        "note": ("Readiness only. No model was downloaded, loaded or executed. "
                 "transformers is present but is not a licence to fetch weights "
                 "for a model this repository has not verified."),
        "candidates": [r.to_dict() for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in results:
        state = "READY" if r.ready else "NOT READY"
        print(f"{state}: {r.model_id}")
        for finding in r.findings:
            mark = "ok  " if finding.ok else "FAIL"
            print(f"  [{mark}] {finding.check}: {finding.detail}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())