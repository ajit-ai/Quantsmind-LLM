"""M2 readiness gate (M2.12).

Every check below reads a real artifact and reports the evidence it found. There
are no hardcoded "pass" results: if an artifact is missing, the check fails, and
if a fact can only come from a human authorisation, the check is BLOCKED rather
than guessed.

Three outcomes are distinguished on purpose:

``PASS``
    The requirement is met, and the evidence is named.
``FAIL``
    A technical deficiency. Someone could fix it by writing code.
``BLOCKED``
    A decision only a human can make: licence interpretation, source
    authorisation, policy. No amount of engineering resolves these.

A milestone whose gate reports BLOCKED is not "nearly done". It is blocked, and
the gate says exactly what is missing.

Usage
-----
    python -m data.m2_gate
    python -m data.m2_gate --json
    python -m data.m2_gate --write data/m2_gate_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"

# Milestone targets. These are the numbers M2 was specified against; the gate
# reports against them rather than against whatever the corpus happens to be.
TARGET_TOKENS = 100_000_000
TARGET_SOURCE_COUNT = 20
TARGET_SPLITS = ("train", "val", "test")

CORPUS = "data/code_corpus_v2.jsonl"
TOKEN_META = "data/code_corpus_v2.tokens.meta.json"
LICENCE_REPORT = "data/licence_verification.json"
RELEASE_DIR = "data/releases/code_corpus-v2"
TRAIN_CONFIG = "configs/train/pretrain_code_v2.yaml"
HOLDOUT = "evaluation/code_holdout_v2.jsonl"
HOLDOUT_MANIFEST = "evaluation/code_holdout_v2.manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Gate:
    def __init__(self):
        self.checks = []

    def add(self, check_id, requirement, status, evidence=None, detail=None,
            remedy=None):
        self.checks.append({
            "id": check_id,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "detail": detail,
            "remedy": remedy,
        })
        return self

    def ok(self, check_id, requirement, evidence):
        return self.add(check_id, requirement, PASS, evidence=evidence)

    def fail(self, check_id, requirement, detail, evidence=None, remedy=None):
        return self.add(check_id, requirement, FAIL, evidence=evidence,
                        detail=detail, remedy=remedy)

    def blocked(self, check_id, requirement, detail, evidence=None, remedy=None):
        return self.add(check_id, requirement, BLOCKED, evidence=evidence,
                        detail=detail, remedy=remedy)

    def counts(self):
        return {
            status: sum(1 for c in self.checks if c["status"] == status)
            for status in (PASS, FAIL, BLOCKED)
        }


def check_schema(gate, corpus_path):
    from data.schema import audit_dataset

    path = Path(corpus_path)
    if not path.exists():
        return gate.fail(
            "M2.4-schema", "Dataset records satisfy the release schema",
            f"corpus not found: {path}",
            remedy="build the corpus before assessing it",
        )
    summary = audit_dataset(path, level="release")
    evidence = (f"{summary['valid']}/{summary['records']} valid, state="
                f"{summary['state']}, kinds={summary['kinds']}")
    if summary["conforms"]:
        return gate.ok("M2.4-schema",
                       "Dataset records satisfy the release schema", evidence)
    return gate.fail(
        "M2.4-schema", "Dataset records satisfy the release schema",
        f"{summary['invalid']} records deviate", evidence=evidence,
        remedy=f"python -m data.schema {path} --level release",
    )


def check_licensing(gate, licence_report_path):
    path = Path(licence_report_path)
    if not path.exists():
        return gate.blocked(
            "M2.3-licence", "Every source has an approved licence",
            f"no licence verification report at {path}",
            remedy="python -m data.verify_licences",
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    evidence = (f"checked={data['counts']['checked']} verified="
                f"{data['counts']['verified']} rejected={data['counts']['rejected']}"
                f" needs_human_review={data['counts']['needs_human_review']}")
    if data.get("all_verified"):
        return gate.ok("M2.3-licence", "Every source has an approved licence",
                       evidence)
    return gate.blocked(
        "M2.3-licence", "Every source has an approved licence",
        f"{data['counts']['needs_human_review']} source(s) need a human licence "
        f"decision: {', '.join(data.get('needs_human_review_ids', []))}",
        evidence=evidence,
        remedy=("A human must decide whether these licences are acceptable for "
                "training data. This is a legal judgement, not a code change."),
    )


def check_source_scale(gate, corpus_path):
    path = Path(corpus_path)
    if not path.exists():
        return gate.fail(
            "M2.3-scale", f"Corpus draws from at least {TARGET_SOURCE_COUNT} sources",
            f"corpus not found: {path}",
        )
    repos = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            repo = json.loads(line).get("repo")
            if repo:
                repos.add(repo)
    evidence = f"{len(repos)} distinct repositories in the corpus"
    if len(repos) >= TARGET_SOURCE_COUNT:
        return gate.ok("M2.3-scale",
                       f"Corpus draws from at least {TARGET_SOURCE_COUNT} sources",
                       evidence)
    return gate.blocked(
        "M2.3-scale", f"Corpus draws from at least {TARGET_SOURCE_COUNT} sources",
        f"only {len(repos)} repositories are present; target is {TARGET_SOURCE_COUNT}. "
        "Expanding the source allowlist requires human authorisation.",
        evidence=evidence,
        remedy="A human must authorise additional sources before they are fetched.",
    )


def check_token_scale(gate, token_meta_path):
    path = Path(token_meta_path)
    if not path.exists():
        return gate.fail(
            "M2.3-tokens", f"Token corpus reaches {TARGET_TOKENS:,} tokens",
            f"no token metadata at {path}",
            remedy="python -m data.encode_dataset --corpus <corpus> ...",
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    tokens = meta.get("num_tokens") or 0
    pct = (tokens / TARGET_TOKENS) * 100
    evidence = (f"{tokens:,} tokens from {meta.get('num_documents')} documents "
                f"({pct:.2f}% of the {TARGET_TOKENS:,} target)")
    if tokens >= TARGET_TOKENS:
        return gate.ok("M2.3-tokens",
                       f"Token corpus reaches {TARGET_TOKENS:,} tokens", evidence)
    return gate.fail(
        "M2.3-tokens", f"Token corpus reaches {TARGET_TOKENS:,} tokens",
        f"{tokens:,} tokens is {TARGET_TOKENS - tokens:,} short of the target",
        evidence=evidence,
        remedy=("Blocked upstream by M2.3-scale: a larger corpus needs more "
                "authorised sources. Adding tokens from the current sources "
                "cannot close a gap this large."),
    )


def check_splits(gate, corpus_path):
    from evaluation.holdout import check_separation

    path = Path(corpus_path)
    if not path.exists():
        return gate.fail("M2.6-splits",
                         "Train/val/test splits do not share repositories",
                         f"corpus not found: {path}")
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    present = {r.get("split") for r in records}
    missing = [s for s in TARGET_SPLITS if s not in present]
    if missing:
        return gate.fail(
            "M2.6-splits", "Train/val/test splits do not share repositories",
            f"missing split(s): {missing}",
        )
    separation = check_separation(records, "test")
    evidence = (f"{separation['holdout_records']} test vs "
                f"{separation['train_records']} train; shared repos="
                f"{separation['shared_repositories']}")
    if separation["ok"]:
        return gate.ok("M2.6-splits",
                       "Train/val/test splits do not share repositories", evidence)
    return gate.fail(
        "M2.6-splits", "Train/val/test splits do not share repositories",
        separation["problem"], evidence=evidence,
    )


def check_release(gate, release_dir):
    from data.release import verify_release

    path = Path(release_dir)
    if not path.exists():
        return gate.fail(
            "M2.8-release", "Released artifacts verify against their checksums",
            f"no release at {path}",
            remedy="python -m data.release build --corpus <corpus> --out <dir>",
        )
    report = verify_release(path)
    passed = [n for n, v in report["checks"].items() if v]
    failed = [n for n, v in report["checks"].items() if not v]
    evidence = (f"{report.get('dataset_id')}: {len(passed)} checks passed, "
                f"{len(failed)} failed")
    if report["ok"]:
        return gate.ok("M2.8-release",
                       "Released artifacts verify against their checksums",
                       evidence)
    return gate.fail(
        "M2.8-release", "Released artifacts verify against their checksums",
        "; ".join(report["problems"]), evidence=evidence,
        remedy=f"python -m data.release verify --release {path}",
    )


def check_training_config(gate, config_path):
    """The config must resolve to a real artifact with a matching vocabulary."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        return gate.fail(
            "M2.9-training",
            "Training config resolves to a real, vocab-matched token artifact",
            f"no config at {path}",
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data_cfg = config.get("data") or {}
    tag = data_cfg.get("dataset_tag")
    requirement = ("Training config resolves to a real, vocab-matched token "
                   "artifact")
    if not tag:
        return gate.fail("M2.9-training", requirement,
                         "config defines no data.dataset_tag", evidence=str(path))

    token_path = Path("data") / f"{str(tag).replace('-', '_')}.tokens.bin"
    if not token_path.exists():
        return gate.fail(
            "M2.9-training", requirement,
            f"dataset_tag {tag!r} resolves to {token_path}, which does not exist",
            evidence=str(path),
            remedy="build the token artifact or correct dataset_tag",
        )

    # Vocabulary agreement: a model whose embedding table is smaller than the
    # tokenizer cannot index its own training data.
    model_cfg = (config.get("run") or {}).get("model_config")
    evidence = f"dataset_tag={tag} -> {token_path}"
    if not model_cfg or not Path(model_cfg).exists():
        return gate.fail(
            "M2.9-training", requirement,
            f"model_config missing or absent: {model_cfg}", evidence=evidence,
        )
    model = yaml.safe_load(Path(model_cfg).read_text(encoding="utf-8")) or {}
    model_vocab = (model.get("model") or model).get("vocab_size")
    meta_path = token_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return gate.fail(
            "M2.9-training", requirement,
            f"token artifact has no metadata: {meta_path}", evidence=evidence,
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizer_vocab = meta.get("vocab_size")
    evidence = (f"{evidence}; {Path(model_cfg).name} vocab_size={model_vocab}, "
                f"tokenizer vocab_size={tokenizer_vocab}")
    if model_vocab is None:
        return gate.fail("M2.9-training", requirement,
                         f"{model_cfg} declares no vocab_size", evidence=evidence)
    if tokenizer_vocab is None:
        return gate.fail("M2.9-training", requirement,
                         "token metadata declares no vocab_size", evidence=evidence)
    if model_vocab < tokenizer_vocab:
        return gate.fail(
            "M2.9-training", requirement,
            f"model vocab_size {model_vocab} is smaller than the tokenizer's "
            f"{tokenizer_vocab}; token ids would be out of range",
            evidence=evidence,
            remedy="raise the model vocab_size or use a matched tokenizer",
        )
    if not meta.get("dataset_schema_validated"):
        return gate.fail(
            "M2.9-training", requirement,
            "token artifact was encoded with dataset schema validation disabled",
            evidence=evidence,
        )
    return gate.ok("M2.9-training", requirement, evidence)


def check_holdout(gate, holdout_path, manifest_path):
    requirement = "Held-out evaluation data exists and is separated from training"
    if not Path(holdout_path).exists():
        return gate.fail(
            "M2.10-holdout", requirement, f"no holdout set at {holdout_path}",
            remedy="python -m evaluation.holdout",
        )
    mpath = Path(manifest_path)
    if not mpath.exists():
        return gate.fail(
            "M2.10-holdout", requirement, f"no holdout manifest at {mpath}",
        )
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    separation = manifest.get("separation") or {}
    protocol = manifest.get("evaluation_protocol") or {}
    evidence = (f"{manifest.get('items')} items across "
                f"{len(manifest.get('repositories', []))} repositories; "
                f"executable_benchmark={protocol.get('executable_benchmark')}")
    if not separation.get("ok"):
        return gate.fail(
            "M2.10-holdout", requirement,
            f"holdout is not separated from training: {separation.get('problem')}",
            evidence=evidence,
        )
    if not protocol.get("limitations"):
        return gate.fail(
            "M2.10-holdout", requirement,
            "holdout manifest records no limitations", evidence=evidence,
        )
    return gate.ok("M2.10-holdout", requirement, evidence)


def check_eval_capability(gate, holdout_manifest_path):
    """Reward an honest manifest, and reject one that overstates capability."""
    requirement = "Evaluation capability is not overstated"
    path = Path(holdout_manifest_path)
    if not path.exists():
        return gate.fail("M2.10-benchmark", requirement,
                         f"no holdout manifest at {path}")
    protocol = (json.loads(path.read_text(encoding="utf-8"))
                .get("evaluation_protocol") or {})
    evidence = (f"published_suites_status={protocol.get('published_suites_status')}, "
                f"executable_benchmark={protocol.get('executable_benchmark')}")
    if protocol.get("executable_benchmark") is False:
        return gate.ok("M2.10-benchmark", requirement, evidence)
    return gate.fail(
        "M2.10-benchmark", requirement,
        "holdout manifest does not declare itself non-executable", evidence=evidence,
    )


def check_human_eval(gate):
    from evaluation.human_eval import RUBRIC

    requirement = "Human evaluation is defined, calibrated and releasable"
    evidence = (f"rubric v{RUBRIC['rubric_version']} with "
                f"{len(RUBRIC['criteria'])} criteria and "
                f"{len(RUBRIC['known_limitations'])} stated limitations")
    return gate.blocked(
        "M2.11-human-eval", requirement,
        "the rubric and protocol are defined, but no annotators have been "
        "recruited and no judgements collected, so no agreement figure exists "
        "and no human score may be released",
        evidence=evidence,
        remedy=("Requires human action: an annotator pool and a judging round. "
                "No code change can produce this evidence."),
    )


def run_gate(corpus=CORPUS, token_meta=TOKEN_META, licence_report=LICENCE_REPORT,
             release_dir=RELEASE_DIR, train_config=TRAIN_CONFIG,
             holdout=HOLDOUT, holdout_manifest=HOLDOUT_MANIFEST):
    gate = Gate()
    check_schema(gate, corpus)
    check_licensing(gate, licence_report)
    check_source_scale(gate, corpus)
    check_token_scale(gate, token_meta)
    check_splits(gate, corpus)
    check_release(gate, release_dir)
    check_training_config(gate, train_config)
    check_holdout(gate, holdout, holdout_manifest)
    check_eval_capability(gate, holdout_manifest)
    check_human_eval(gate)

    counts = gate.counts()
    if counts[FAIL] or counts[BLOCKED]:
        verdict = "M2 BLOCKED"
        summary = (
            f"M2 is not complete: {counts[FAIL]} technical failure(s), "
            f"{counts[BLOCKED]} blocked on human decisions. "
            "Blocked items cannot be closed by writing code."
        )
    else:
        verdict = "M2 READY"
        summary = "All M2 readiness checks pass with recorded evidence."
    return {
        "schema": "quantsmind.m2_gate/1",
        "generated_at": utc_now_iso(),
        "verdict": verdict,
        "summary": summary,
        "counts": counts,
        "checks": gate.checks,
        "human_decisions_required": [
            {"id": c["id"], "requirement": c["requirement"], "detail": c["detail"]}
            for c in gate.checks if c["status"] == BLOCKED
        ],
        "technical_failures": [
            {"id": c["id"], "requirement": c["requirement"], "detail": c["detail"]}
            for c in gate.checks if c["status"] == FAIL
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M2 readiness gate")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--token-meta", default=TOKEN_META)
    parser.add_argument("--licence-report", default=LICENCE_REPORT)
    parser.add_argument("--release", default=RELEASE_DIR)
    parser.add_argument("--train-config", default=TRAIN_CONFIG)
    parser.add_argument("--holdout", default=HOLDOUT)
    parser.add_argument("--holdout-manifest", default=HOLDOUT_MANIFEST)
    parser.add_argument("--write", default=None, help="write the report as JSON")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args(argv)

    report = run_gate(
        corpus=args.corpus, token_meta=args.token_meta,
        licence_report=args.licence_report, release_dir=args.release,
        train_config=args.train_config, holdout=args.holdout,
        holdout_manifest=args.holdout_manifest,
    )

    if args.write:
        Path(args.write).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["verdict"] == "M2 READY" else 1

    print(f"M2 readiness gate")
    print(f"  verdict : {report['verdict']}")
    print(f"  summary : {report['summary']}")
    print()
    for check in report["checks"]:
        mark = {PASS: "PASS", FAIL: "FAIL", BLOCKED: "BLOCK"}[check["status"]]
        print(f"  [{mark:4}] {check['id']}: {check['requirement']}")
        if check.get("evidence"):
            print(f"          evidence: {check['evidence']}")
        if check.get("detail"):
            print(f"          detail:   {check['detail']}")
        if check.get("remedy"):
            print(f"          remedy:   {check['remedy']}")
    if report["human_decisions_required"]:
        print()
        print("  Human decisions required to close M2:")
        for item in report["human_decisions_required"]:
            print(f"    - {item['id']}: {item['detail']}")
    return 0 if report["verdict"] == "M2 READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())