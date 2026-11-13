"""TAKSH L0.5 — WP2: training-data feasibility model.

Computes unique-token requirements for each model scenario x token:parameter
ratio, against the *measured* licensed supply in this repository, and registers
candidate external datasets with their license/access/legal status. Nothing is
downloaded; no candidate is ingested. The corpus supply figure comes from
configs/feasibility/l05_scenarios.yaml (measured upstream).

Writes reports/l05_data_feasibility.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCENARIOS_PATH = REPO / "configs" / "feasibility" / "l05_scenarios.yaml"


def _yaml():
    import yaml

    return yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))


# Curated register of candidate corpora. Assessment only; NOTHING here was
# acquired or ingested during L0.5. License fields distinguish the dataset's
# access terms from the original source files' licenses, which remain in
# effect per-file and govern copying/adaptation/redistribution.
DATASET_REGISTER = [
    {
        "name": "BigCode The Stack (v1)",
        "official_source": "bigcode-project.github.io (BigCode, Hugging Face)",
        "languages": "358 programming languages incl. Python, JS/TS, Java, C/C++, Go, Rust",
        "published_size": "reported as ~6.4 TB across ~200M repositories' files; filtered release is smaller (reported ~1 TB)",
        "access": "registration + terms acceptance on the Hugging Face hub",
        "license": "BigCode OpenRAIL License (data access); per-file licenses follow the originating repositories (MIT/Apache-2.0/BSD/CC0-class filtered variants exist)",
        "source_code_obligations": "copyleft files (GPL/LGPL) in bulk subsets impose source-offer/notice obligations; BigCode provides a license-filtered subset precisely to avoid this",
        "attribution": "per-file origin must be tracked for attribution if redistribution is ever contemplated",
        "redistribution": "not for public redistribution of the corpus itself; derivative models permitted with naming/notice requirements under OpenRAIL variants",
        "opt_out": "opt-out mechanism exists upstream; downstream users inherit a duty to honour removal requests",
        "commercial_use": "BigCode OpenRAIL permits commercial use subject to use restrictions; per-file copyleft can complicate proprietary redistribution",
        "quality_security": "large-scale secret contamination reported; requires credential scanning; near-duplicate heavy; documentation quality uneven",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "STRONG candidate pending legal review; biggest realistic source of Python tokens",
    },
    {
        "name": "BigCode The Stack v2 / StarCoder2 data",
        "official_source": "bigcode-project.github.io (StarCoder2 / The Stack v2)",
        "languages": "619 languages (reported); Python ~15% of content",
        "published_size": "reported 67.5 TB raw; tokenized code corpora ~2-3T tokens (reported)",
        "access": "registration + terms acceptance",
        "license": "BigCode OpenRAIL-M (M = for models); per-file filtering to permissive/public licenses applied by upstream",
        "source_code_obligations": "subset retains only files whose licenses permit codebase inclusion; per-file origin tracking required",
        "attribution": "expected on any redistribution of the corpus; derivative model notices advised",
        "redistribution": "not redistributable as a corpus; model outputs not covered by corpus license",
        "opt_out": "opt-out honoured; users agreed to honour it downstream",
        "commercial_use": "restricted use categories (no large-scale surveillance, no weapons etc.); commercial generally permitted",
        "quality_security": "license-filtered and deduplicated upstream; still needs secret scan + near-dup and benchmark-leakage re-check by us",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "STRONG candidate; best-aligned with this project's filtering discipline",
    },
    {
        "name": "CodeSearchNet (GitHub-derived)",
        "official_source": "github.com/github/CodeSearchNet",
        "languages": "Go, Java, JavaScript, PHP, Python, Ruby",
        "published_size": "~6M functions with docstrings (reported), ~2 GB",
        "access": "public download, no registration",
        "license": "dataset metadata CC-BY-4.0; the code files carry their ORIGINAL licenses (GitHub ToS apply); not uniformly permissive",
        "source_code_obligations": "GPL/LGPL files inside it impose obligations; GitHub ToS restricts scraping-scale redistribution",
        "attribution": "required if redistributed",
        "redistribution": "prohibited at scale under GitHub ToS without API-level compliance",
        "opt_out": "none provided by upstream",
        "commercial_use": "ambiguous per-file; cannot be assumed",
        "quality_security": "docstring-paired functions (useful for NL-code pairs) but mixed licensing",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "useful shape, licensing too heterogeneous to use as-is",
    },
    {
        "name": "PyPI permissively-licensed packages (direct package index)",
        "official_source": "pypi.org / files.pythonhosted.org",
        "languages": "Python",
        "published_size": "package count > 500k (reported); code volume ~1-2 TB raw, ~tens of billions of tokens",
        "access": "public index, no registration (fair-use limits apply at scrape scale)",
        "license": "per-package license files (majority MIT/Apache/BSD/PSF; copyleft minority must be filtered)",
        "source_code_obligations": "each package's LICENSE file governs; must be read and filtered at package level",
        "attribution": "per-package attribution required for redistribution",
        "redistribution": "per-package; systematic filtering required (SPDX id + allowlist)",
        "opt_out": "none systematic",
        "commercial_use": "mostly permissive but cannot be assumed per-package without reading LICENSEs",
        "quality_security": "uneven quality; secret leakage common (CI tokens, keys); heavy dedup required; documentation/example code high value",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "VIABLE via the same license-gate tooling used for M3-verified; highest legal-engineering cost",
    },
    {
        "name": "RedPajama / Dolma corpora",
        "official_source": "github.com/togethercomputer/RedPajama-Data; allenai/OLMo-Dolma",
        "languages": "many incl. code (RedPajama-Data-1B has code slice), English-dominant",
        "published_size": "1.2 T tokens (reported) for RedPajama-1; Dolma ~3 T",
        "access": "public downloads",
        "license": "mixed CommonCrawl licenses + subset-specific licenses (e.g. C4: ODC-BY); code slice inherits GitHub licensing",
        "source_code_obligations": "same per-file license issue as GitHub-derived code",
        "attribution": "required",
        "redistribution": "corpus redistribution prohibited; derivative research use permitted",
        "opt_out": "none",
        "commercial_use": "allowed for research under ODC-BY-1.0-style terms with attribution; code slice inherits mixed terms",
        "quality_security": "heavy filtering applied; still needs code-specific quality pass",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "NOT a code-first option; only if chat/general text is desired",
    },
    {
        "name": "The Pile",
        "official_source": "github.com/EleutherAI/pile; Hugging Face",
        "languages": "many; code slice ~9% (BigCode subset, GPL-style licensing inside)",
        "published_size": "~800 GB raw (reported), ~1.2T tokens est.",
        "access": "public download",
        "license": "mixed: CC-BY-SA-3.0 (some slices), ArXiv (non-commercial-like terms in practice), GPL-family code slice; license indeterminate per-file",
        "source_code_obligations": "code slice inherits GPL/copyleft obligations; attribution and source-offer very complex",
        "attribution": "complex per-slice",
        "redistribution": "practically unworkable as a redistributed corpus; research use common",
        "opt_out": "none",
        "commercial_use": "generally NOT clean for commercial use of the full corpus; per-slice analysis mandatory",
        "quality_security": "large-scale, mixed provenance; known PII/malware content documented by authors",
        "legal_review_required": True,
        "ingested_in_l05": False,
        "verdict": "EXCLUDE for code training until per-slice review; not recommended",
    },
    {
        "name": "OASST2 (Open Assistant) instruction trees",
        "official_source": "huggingface.co/datasets/OpenAssistant/oasst2",
        "languages": "English + European languages",
        "published_size": "~90k annotated messages (reported)",
        "access": "CC-BY-SA-4.0 (human-authored, Apache-2.0-ish), public download",
        "license": "CC-BY-SA-4.0 (share-alike)",
        "source_code_obligations": "share-alike applies to adapted data, not to models trained on it (per OpenRAIL-style readings), but legal review advised; code content is thin",
        "attribution": "required",
        "redistribution": "permitted with attribution, share-alike on the dataset itself",
        "opt_out": "implicit (contributor consent flow); small dataset, low value for code",
        "commercial_use": "permitted under CC-BY-SA-4.0 with attribution",
        "quality_security": "small, high-quality human data; not code-heavy",
        "legal_review_required": False,
        "ingested_in_l05": False,
        "verdict": "small CC-BY-SA general/instruction data; insufficient code mass",
    },
    {
        "name": "Databricks Dolly 15k",
        "official_source": "huggingface.co/datasets/databricks/databricks-dolly-15k",
        "languages": "English",
        "published_size": "15,187 records (reported)",
        "access": "CC-BY-SA-3.0 public",
        "license": "CC-BY-SA-3.0",
        "source_code_obligations": "share-alike on data; code segment tiny",
        "attribution": "required",
        "redistribution": "permitted with attribution",
        "opt_out": "none systematic; human-authored, low risk",
        "commercial_use": "permitted with attribution",
        "quality_security": "human-authored but tiny and general; near-zero code",
        "legal_review_required": False,
        "ingested_in_l05": False,
        "verdict": "instruction-format donor only; negligible for code capability",
    },
    {
        "name": "Wiki / StackExchange dumps (general text)",
        "official_source": "wikimedia dumps; archive.org SE dumps",
        "languages": "English + others",
        "published_size": "Wikipedia ~20 GB compressed text (reported); SE ~100 GB",
        "access": "public dumps, no registration",
        "license": "Wikipedia CC-BY-SA-3.0; StackExchange dumps CC-BY-SA-4.0",
        "source_code_obligations": "share-alike on data; code inside posts inherits mixed per-post licensing (StackOverflow CC-BY-SA-4.0 with attribution)",
        "attribution": "required",
        "redistribution": "permitted with attribution, share-alike on adapted data",
        "opt_out": "implicit via community conventions; large corpus, needs PII/toxicity filtering",
        "commercial_use": "permitted with attribution",
        "quality_security": "StackOverflow code is high quality; license attribution burden real",
        "legal_review_required": False,
        "ingested_in_l05": False,
        "verdict": "acceptable general-text; code volume still small relative to need",
    },
    {
        "name": "Evaluation benchmarks as training data (HumanEval / MBPP / EvalPlus / L0.2 bench)",
        "official_source": "openai/human-eval; google-research MBPP; evalplus; this repo",
        "languages": "Python",
        "published_size": "164 / 974 / ~1,000+ problems",
        "access": "public",
        "license": "HumanEval MIT; MBPP CC-BY-4.0; EvalPlus MIT",
        "source_code_obligations": "n/a (too small; not a corpus)",
        "attribution": "required if used",
        "redistribution": "permitted",
        "opt_out": "n/a",
        "commercial_use": "permitted",
        "quality_security": "THE point: these are evaluation sets; ANY training on them destroys the measurement",
        "legal_review_required": False,
        "ingested_in_l05": False,
        "verdict": "EXCLUDED BY DESIGN AND BY PROJECT POLICY (no training on eval sets)",
    },
]


def grid(scenarios, supply):
    code = supply["code_tokens"]
    l04 = supply["total_l04_train"]
    l03 = supply["general_tokens_l03"]
    rows = []
    params_by_key = {
        "current": 28_787_328,
        "A": 110_658_816,
        "B": 289_115_136,
        "C": 1_018_624_000,
    }
    for name, cfg in scenarios["architecture_scenarios"].items():
        params = params_by_key[name]
        for ratio in scenarios["token_ratios"]:
            required = params * ratio
            # qualified supply = licence-verified code in-repo
            code_supply = code
            general_supply = l03 + l04 - code  # general text available (CC-BY-SA/PD)
            best_mix_supply = code_supply + general_supply
            rows.append({
                "scenario": name,
                "params": params,
                "tokens_per_param": ratio,
                "required_unique_tokens": required,
                "qualified_code_supply": code_supply,
                "qualified_total_supply": best_mix_supply,
                "code_gap_tokens": required - code_supply,
                "code_supply_coverage": round(code_supply / required, 6),
                "best_mix_coverage": round(best_mix_supply / required, 6),
                "epochs_code_only": round(required / code_supply, 1),
                "storage_bytes_uint16": required * 2,
                "storage_gb_uint16": round(required * 2 / 1e9, 3),
                "note": "storage assumes uint16 flat token stream + ~5% metadata margin not included",
            })
    return rows


def mixture_guidance(params: int) -> str:
    return (
        "practical mixture for a code model: aim >= 60% high-quality code tokens, "
        "bounded general-text fraction (~20-30%) for instruction following; "
        "repetition beyond ~4 epochs of unique data yields strongly diminished returns"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l05_data_feasibility.json")
    args = parser.parse_args(argv)

    scenarios = _yaml()
    supply = scenarios["qualified_token_supply"]
    rows = grid(scenarios, supply)

    out = {
        "schema": "taksh.l05_data_feasibility/1",
        "generated_by": "evaluation/l05_data_feasibility.py",
        "token_supply": supply,
        "token_supply_disclosure": [
            "code supply (1,590,703) is licence-verified in-repo code (M3 MIT/BSD + Apache-2.0 repo snapshot).",
            "general supply (l03 ~19.6M) is English Wikipedia/Gutenberg (CC-BY-SA-3.0 / Public Domain); it is not code.",
            "corpus_v0's 9,993,983-token figure uses the obsolete 49,152-vocab tokenizer; re-encoded equivalent would be ~9-11M tokens.",
            "counting repeated epochs as new data is not done anywhere in this analysis.",
        ],
        "requirements_grid": rows,
        "mixture_guidance": mixture_guidance(0),
        "dataset_candidate_register": DATASET_REGISTER,
        "register_summary": {
            "candidates": len(DATASET_REGISTER),
            "legal_review_required": sum(1 for d in DATASET_REGISTER if d["legal_review_required"]),
            "zero_ingested_in_l05": all(not d["ingested_in_l05"] for d in DATASET_REGISTER),
            "policy": "No candidate may enter training merely because it is publicly accessible; every candidate above requires legal sign-off before acquisition or ingestion.",
        },
        "census": {
            "code_tokens_in_repo": supply["code_tokens"],
            "epochs_for_1B_at_100_per_param": round(
                1_018_624_000 * 100 / supply["code_tokens"], 1
            ),
            "epochs_for_300M_at_100_per_param": round(
                289_115_136 * 100 / supply["code_tokens"], 1
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"requirements grid: {len(rows)} scenarios")
    print(f"code supply coverage, 1B@100tok/param: {rows[-1]['code_supply_coverage']:.4%}")
    print(f"register: {out['register_summary']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
