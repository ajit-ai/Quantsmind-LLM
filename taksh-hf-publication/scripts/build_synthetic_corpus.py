"""Build a synthetic qualification corpus for Taksh v1.3.0.

The corpus is generated locally, from a fixed seed, and is explicitly
labelled ``dataset_class: synthetic``. It exists so the training pipeline can
be qualified on a controlled distribution without a licensed corpus.

It is not a corpus. It is not approved for anything, it is not a sample of
real Python, and the training authorization gate refuses to treat it as
production data.

Usage
-----
    python scripts/build_synthetic_corpus.py --out runs/v13_synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.python_v1.pipeline import build_dataset
from data.python_v1.registry import snapshot_digest

LICENCE_NOTICE = (
    "Synthetic qualification fixture. Generated locally by\n"
    "scripts/build_synthetic_corpus.py. Not third-party content.\n"
)
SPDX = "LicenseRef-Taksh-Synthetic"

WORDS = (
    "buffer cursor ledger packet segment index token window frame block "
    "record stream channel queue bundle shard matrix vector offset stride"
).split()


def _module(seed: int, index: int) -> str:
    rng = random.Random(seed * 1000 + index)
    names = rng.sample(WORDS, 8)
    lines = [
        f'"""Synthetic module {index} for qualification only."""',
        "",
        "import os",
        "import sys",
        "",
    ]
    for position, name in enumerate(names):
        lines += [
            f"def {name}_{index}(value):",
            f'    """Compute the {name} for module {index}."""',
            f"    scaled = value * {position + 2} + {index}",
            "    total = scaled + len(os.sep) + len(os.sep)",
            "    return total",
            "",
        ]
    lines += [
        f"def summarize_{index}(rows):",
        f'    """Summarize rows for module {index}."""',
        "    count = 0",
        "    for row in rows:",
        "        count += len(str(row))",
        "    return count",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_corpus(root: Path, sources: int, modules: int, seed: int) -> list:
    """Create the synthetic source tree and return the contract entries."""
    entries = []
    for source_index in range(sources):
        source_dir = root / f"src_{source_index:02d}"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "LICENSE").write_text(LICENCE_NOTICE, encoding="utf-8")
        for module_index in range(modules):
            (source_dir / f"mod_{module_index:02d}.py").write_text(
                _module(seed, source_index * 100 + module_index), encoding="utf-8"
            )
        snapshot = []
        for path in sorted(source_dir.rglob("*.py")):
            snapshot.append(
                (
                    path.relative_to(source_dir).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        entries.append(
            {
                "id": f"synthetic_src_{source_index:02d}",
                "kind": "local_directory",
                "location": source_dir.name,
                "revision": snapshot_digest(snapshot),
                "project": f"synthetic_project_{source_index % max(1, sources // 3):02d}",
                "attribution": "Synthetic qualification fixture",
                "allowed_extensions": [".py"],
                "excluded_paths": [],
                "license": {
                    "spdx": SPDX,
                    "decision": "approved",
                    "evidence": "LICENSE",
                    "evidence_sha256": hashlib.sha256(
                        (source_dir / "LICENSE").read_bytes()
                    ).hexdigest(),
                },
                "approval": {
                    "status": "approved",
                    "reviewer": "synthetic-fixture",
                    "rationale": "locally generated qualification fixture",
                },
            }
        )
    return entries


def _contract(entries, seed: int) -> dict:
    return {
        "schema": "quantsmind.python_data_contract/1",
        "version": "synthetic_v13",
        "dataset_class": "synthetic",
        "policy": {
            "default_decision": "deny",
            "allowed_spdx": [SPDX],
            "require_pinned_revision": True,
            "require_license_evidence": True,
        },
        "limits": {
            "max_files": 1000,
            "max_total_bytes": 50_000_000,
            "max_file_bytes": 200_000,
            "max_path_depth": 40,
        },
        "encoding": {"primary": "utf-8", "fallbacks": [], "allow_bom": True},
        "normalization": {"newline": "\n", "strip_bom": True},
        "quality": {
            "min_bytes": 200,
            "max_bytes": 200_000,
            "min_lines": 5,
            "min_unique_token_ratio": 0.10,
            "minified_line_length": 400,
            "minified_average_line_length": 200,
            "max_compression_ratio": 0.08,
            "generated_markers": ["@generated", "do not edit"],
            "excluded_path_patterns": ["(^|/)(vendor|tests?)(/|$)"],
        },
        "dedup": {
            "exact": True,
            "near": True,
            "shingle_size": 8,
            "minhash_perms": 64,
            "minhash_bands": 16,
            "threshold": 0.85,
            "boilerplate_df_ratio": 0.5,
            "max_shingles_per_record": 4000,
            "max_bucket_candidates": 64,
            "group_by_source": True,
            "seed": 120,
        },
        "splits": {
            "proportions": {"train": 0.7, "val": 0.15, "test": 0.15},
            "seed": seed,
            "min_groups_per_split": 1,
            "min_records_per_split": 1,
        },
        "benchmarks": {"registry": "benchmarks.yaml", "ngram_size": 13},
        "tokenizer": {"path": "tokenizer/qm_code_bpe_v0.json"},
        "sources": entries,
    }


def _train_config(root: Path) -> dict:
    """A bounded synthetic-mode training configuration next to the corpus."""
    return {
        "schema": "quantsmind.train_config/1",
        "mode": "synthetic",
        "run": {
            "name": "taksh-v13-synthetic",
            "model_config": "configs/model/code_m3_9m.yaml",
            "seed": 20261021,
        },
        "data": {
            "manifest": "corpus/manifest.json",
            "tokenizer": "tokenizer/qm_code_bpe_v0.json",
            "context_length": 256,
            "sequence_policy": "document",
            "append_eos": True,
            "allow_cross_document": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr_peak": 3.0e-3,
            "warmup_steps": 5,
            "min_lr_ratio": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
        },
        "batch": {
            "micro_batch_size": 2,
            "grad_accum_steps": 2,
            "shuffle": True,
        },
        "precision": {"device": "cpu", "dtype": "float32"},
        "training": {
            "max_steps": 40,
            "max_tokens": 200000,
            "max_wall_seconds": 600,
            "checkpoint_every_steps": 20,
            "eval_every_steps": 10,
            "val_batches": 4,
            "output_dir": "train_out",
        },
        "limits": {"max_checkpoints": 2, "max_output_bytes": 200000000},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic qualification corpus and its manifest"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", type=int, default=12)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = build_corpus(root, args.sources, args.modules, args.seed)

    contract_path = root / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(_contract(entries, args.seed), sort_keys=True), encoding="utf-8"
    )
    (root / "benchmarks.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "quantsmind.benchmark_registry/1",
                "version": "synthetic",
                "suites": [
                    {
                        "id": "none_available",
                        "version": "0",
                        "availability": "unavailable",
                        "notes": "no local benchmark copy; nothing was checked",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = build_dataset(contract_path, root / "corpus")
    (root / "pretrain_v13_synthetic.yaml").write_text(
        yaml.safe_dump(_train_config(root), sort_keys=True), encoding="utf-8"
    )
    print(f"corpus_class   : {manifest['notes']['corpus_class']}")
    print(f"synthetic      : {manifest['notes']['synthetic']}")
    print(f"corpus_ready   : {manifest['notes']['corpus_ready']}")
    print(f"sources approved: {manifest['sources']['approved']}")
    print(f"samples kept   : {manifest['filtering']['kept']}")
    print(f"splits         : {manifest['splits']['counts']} ready={manifest['splits']['ready']}")
    print(f"manifest       : {root / 'corpus' / 'manifest.json'}")
    print(f"manifest hash  : {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
