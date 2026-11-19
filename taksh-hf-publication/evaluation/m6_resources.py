"""M6 resource preflight: conservative memory and disk estimation.

The point of this module is to *refuse* to load a model that would exhaust the
machine. Estimates are deliberately pessimistic and every one of them is
labelled ``ESTIMATE`` so a later report can never present them as measurements.

Reuses ``training.preflight`` for real platform measurements, so M6 does not
grow a second divergent notion of "how much memory does this box have".

    python -m evaluation.m6_resources --model-id some/model --parameters 1000000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training.preflight import (disk_space, peak_rss_bytes, system_memory,
                                torch_facts)

GIB = 1024 ** 3
MIB = 1024 ** 2

# Bytes per parameter by dtype. Used to size a candidate whose parameter count
# is known but whose artifact has not been measured.
BYTES_PER_PARAM = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float8": 1, "fp8": 1, "int8": 1,
    "int4": 0.5, "nf4": 0.5, "q4": 0.5,
}

ACTIVATION_FACTOR = 1.25         # transient buffers and the forward pass
KV_CACHE_BYTES_PER_TOKEN = 2     # one for K, one for V
OS_HEADROOM_BYTES = 512 * MIB    # the OS still needs RAM while we run
DISK_HEADROOM_BYTES = 2 * GIB    # never fill the volume to the last byte


@dataclass
class Requirement:
    """One line of an estimate, with the basis for the number attached."""

    component: str
    bytes: int
    basis: str
    measured: bool = False

    @property
    def mib(self) -> str:
        return f"{self.bytes / MIB:.0f} MiB"


@dataclass
class ResourcePlan:
    """A conservative estimate of what loading and using a model will cost."""

    model_id: str
    requirements: list = field(default_factory=list)
    available_ram_bytes: int | None = None
    total_ram_bytes: int | None = None
    available_disk_bytes: int | None = None
    required_disk_bytes: int = 0
    estimate_confidence: str = "COARSE"

    @property
    def required_ram_bytes(self) -> int:
        return sum(r.bytes for r in self.requirements)

    @property
    def fits_ram(self) -> bool:
        if not self.available_ram_bytes:
            return True
        return self.required_ram_bytes <= self.available_ram_bytes

    @property
    def fits_disk(self) -> bool:
        if not self.available_disk_bytes:
            return True
        return self.required_disk_bytes <= self.available_disk_bytes

    @property
    def ok(self) -> bool:
        return self.fits_ram and self.fits_disk

    def decision(self) -> tuple[bool, str]:
        """Preflight verdict plus a reason a human can act on."""
        caveat = ("" if self.estimate_confidence == "MEASURED_GEOMETRY"
                  else " [estimate confidence: COARSE - model config not read]")
        if not self.fits_ram:
            return False, (
                f"insufficient RAM: conservative estimate is "
                f"{self.required_ram_bytes / GIB:.2f} GiB but only "
                f"{(self.available_ram_bytes or 0) / GIB:.2f} GiB is available; "
                f"refusing to load{caveat}"
            )
        if not self.fits_disk:
            return False, (
                f"insufficient disk: {self.required_disk_bytes / GIB:.2f} GiB "
                f"required, "
                f"{(self.available_disk_bytes or 0) / GIB:.2f} GiB available"
            )
        return True, (
            f"estimated {self.required_ram_bytes / MIB:.0f} MiB RAM against "
            f"{(self.available_ram_bytes or 0) / MIB:.0f} MiB available "
            f"(host totals measured; all other figures are ESTIMATE){caveat}"
        )

    def to_dict(self) -> dict:
        ok, reason = self.decision()
        return {
            "model_id": self.model_id,
            "required_ram_bytes": self.required_ram_bytes,
            "required_ram_gib": round(self.required_ram_bytes / GIB, 3),
            "available_ram_gib": (round(self.available_ram_bytes / GIB, 3)
                                  if self.available_ram_bytes else None),
            "total_ram_gib": (round(self.total_ram_bytes / GIB, 3)
                              if self.total_ram_bytes else None),
            "required_disk_gib": round(self.required_disk_bytes / GIB, 3),
            "available_disk_gib": (round(self.available_disk_bytes / GIB, 3)
                                   if self.available_disk_bytes else None),
            "fits_ram": self.fits_ram,
            "fits_disk": self.fits_disk,
            "ok": ok,
            "reason": reason,
            "requirements": [asdict(r) | {"mib": r.mib}
                             for r in self.requirements],
            "estimate_kind": "ESTIMATE",
        }


def kv_cache_bytes(parameters: int, context: int, *, layers=None,
                   kv_heads=None, head_dim=None, kv_ratio: int = 1
                   ) -> tuple[int, str, str]:
    """KV cache for a single sequence.

    Returns ``(bytes, basis, confidence)``.

    ``confidence`` is ``MEASURED_GEOMETRY`` when the candidate's config supplied
    real layer/head counts, and ``COARSE`` when the value had to be inferred.
    Preflight reports the confidence so that a reader can tell a trustworthy
    number from a rough one; ``m6_resources`` refuses to present a coarse
    estimate as if it were exact.

    The exact form is ``2 (K and V) * layers * kv_heads * head_dim * 2 bytes``.
    Without geometry there is no way to recover ``layers`` from a parameter
    count alone -- the relation ``parameters ~= 12 * layers * hidden_size`` still
    contains it -- so the fallback assumes a conventional 32 layers and says so.
    """
    context = max(1, context)
    if layers and kv_heads and head_dim:
        per_token = 2 * layers * kv_heads * head_dim * KV_CACHE_BYTES_PER_TOKEN
        basis = (f"{layers} layers x {kv_heads} kv-heads x {head_dim} head-dim "
                 f"from the model config")
        confidence = "MEASURED_GEOMETRY"
    else:
        assumed_layers = 32
        # params_body ~= 12 * layers * hidden_size**2 for a dense transformer,
        # hence hidden = sqrt(params / (12 * layers)). Using 12*L*H (missing the
        # square) overestimates hidden size by two orders of magnitude.
        hidden = (parameters / (12.0 * assumed_layers)) ** 0.5
        per_token = 2 * assumed_layers * hidden * KV_CACHE_BYTES_PER_TOKEN
        basis = (f"COARSE: assumed {assumed_layers} layers from a conventional "
                 f"depth (config geometry not read)")
        confidence = "COARSE"
    return int(per_token * context), basis, confidence


def plan_for(model_id, parameters=None, *, dtype="bfloat16", context=512,
             weights_bytes=None, artifact_bytes=None, extra_bytes=0,
             layers=None, kv_heads=None, head_dim=None) -> ResourcePlan:
    """Build a conservative plan.

    ``parameters`` is used when known. Otherwise the measured artifact size is
    the basis, which is precisely why a disk measurement is never treated as a
    total runtime requirement: activations and KV cache are added on top.

    ``layers``/``kv_heads``/``head_dim`` come from the candidate's config when
    it has been read, and sharpen the KV-cache estimate considerably.
    """
    bpp = BYTES_PER_PARAM.get(str(dtype).lower(), 2)
    plan = ResourcePlan(model_id=model_id)

    if weights_bytes is not None:
        plan.requirements.append(Requirement(
            "weights", int(weights_bytes),
            f"measured artifact size on disk ({dtype})", measured=True))
    elif parameters is not None:
        plan.requirements.append(Requirement(
            "weights", int(parameters * bpp),
            f"ESTIMATE: {parameters:,} parameters x {bpp} bytes ({dtype})"))
    else:
        plan.requirements.append(Requirement(
            "weights", 0,
            "unknown: neither a parameter count nor an artifact size was given"))

    weights = plan.requirements[0].bytes
    params_for_kv = parameters or (int(weights / bpp) if weights else 0)
    kv_bytes, kv_basis, kv_conf = kv_cache_bytes(
        params_for_kv, context, layers=layers, kv_heads=kv_heads,
        head_dim=head_dim)
    plan.requirements.append(Requirement(
        "kv_cache", kv_bytes, f"ESTIMATE [{kv_conf}]: {kv_basis}, "
        f"{context}-token context"))
    plan.estimate_confidence = (
        "MEASURED_GEOMETRY" if kv_conf == "MEASURED_GEOMETRY" else "COARSE")
    plan.requirements.append(Requirement(
        "activations_and_buffers", int(weights * (ACTIVATION_FACTOR - 1)),
        f"ESTIMATE: {ACTIVATION_FACTOR:.2f}x weight footprint for transient "
        f"buffers and the forward pass"))
    if extra_bytes:
        plan.requirements.append(Requirement(
            "runtime_overhead", int(extra_bytes),
            "ESTIMATE: interpreter, framework and tokenizer overhead"))

    memory = system_memory()
    plan.total_ram_bytes = memory.get("total_bytes")
    plan.available_ram_bytes = memory.get("available_bytes")
    plan.available_disk_bytes = disk_space(".").get("free_bytes")
    plan.required_disk_bytes = int((artifact_bytes or weights)
                                   + DISK_HEADROOM_BYTES)
    return plan


def machine_facts() -> dict:
    """Measured host facts. Unlike the rest of this module, these are real."""
    memory = system_memory()
    disk = disk_space(".")
    return {
        "measured": True,
        "total_ram_gib": round((memory.get("total_bytes") or 0) / GIB, 3),
        "available_ram_gib": round((memory.get("available_bytes") or 0) / GIB, 3),
        "disk_free_gib": round((disk.get("free_bytes") or 0) / GIB, 3),
        "peak_rss_mib": round((peak_rss_bytes() or 0) / MIB, 1),
        "torch": torch_facts(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M6 resource preflight for a candidate model")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--parameters", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--artifact", default=None,
                        help="measure this file instead of assuming a size")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    size = Path(args.artifact).stat().st_size if args.artifact else None
    plan = plan_for(args.model_id, args.parameters, dtype=args.dtype,
                    context=args.context, weights_bytes=size, artifact_bytes=size)
    payload = {"machine": machine_facts(), "plan": plan.to_dict()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("machine (measured):")
    print(json.dumps(payload["machine"], indent=2))
    ok, reason = plan.decision()
    print(f"\n{args.model_id}: {'OK' if ok else 'BLOCKED'} - {reason}")
    for req in plan.requirements:
        kind = "measured" if req.measured else "ESTIMATE"
        print(f"  {req.component:24} {req.mib:>12}  [{kind}] {req.basis}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())