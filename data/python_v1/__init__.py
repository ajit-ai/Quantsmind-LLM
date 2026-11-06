"""Python data foundation for Taksh v1.2.0.

The package implements the v1.2.0 pipeline stages as independent, testable
modules:

``registry``
    Default-deny source registry: identity, pinned revision, licence evidence,
    owner approval.
``ingest``
    Bounded, read-only, symlink-safe local traversal and decoding.
``quality``
    Normalisation, AST validation, quality rules and secret detection.
``dedup``
    Deterministic exact and bounded near-duplicate detection with groups.
``splits``
    Source-group and duplicate-group aware train/val/test assignment.
``benchmarks``
    Versioned benchmark registry and contamination screening.
``manifest``
    Deterministic manifest, artifact checksums and data card.
``interface``
    Validated read interface for the v1.3.0 training pipeline.
``pipeline``
    Orchestration and the ``python -m data.python_v1`` command.

Data preparation is deliberately independent of model training: nothing here
imports a model, and no stage performs network access.
"""

PIPELINE_VERSION = "python-v1.2.0"
CONTRACT_SCHEMA = "quantsmind.python_data_contract/1"
RECORD_SCHEMA = "quantsmind.python_record/1"
MANIFEST_SCHEMA = "quantsmind.python_dataset_manifest/1"
BENCHMARK_SCHEMA = "quantsmind.benchmark_registry/1"
