# TAKSH — Local Capability Corpus Provenance

**Date:** 2026-11-20
**Status:** local experimental corpus only. Not a public dataset. Not licence-cleared for redistribution.
**Builder:** `scripts/build_capability_corpus.py`

---

## 1. Constraints this corpus obeys

The owner authorised a capability experiment with three hard constraints:

1. **No money.**
2. **No external data acquisition.**
3. **CPU-only, bounded local compute.**

The corpus therefore uses only material already present on this machine:

- this repository's own committed Python source (Apache-2.0);
- installed open-source packages in `taksh-env/Lib/site-packages` (read-only).

No network request, download, purchase, or external dataset is involved.

## 2. Composition

| Item | Value |
|---|---|
| Records (chunks) | 6,973 |
| Source files yielding records | 6,790 |
| Repository sources | 172 |
| Installed-package sources | 6,618 |
| JSONL size | 39.6 MiB |
| Encoded tokens | 13,661,071 |
| Tokenizer | `tokenizer/taksh_code_bpe_v1.json` (vocab 5,822) |
| Corpus SHA-256 | `0052ed153a2c765dede6a68b193e17ebdfc33951f99f2c9839bc0dc980d9b273` |
| Manifest | `data/cap_corpus.manifest.json` |

The encoded token file (`data/cap_corpus.tokens.bin`) is gitignored and stays local.

## 3. Licence inventory of installed sources

The installed environment contains permissive licences plus three MPL-2.0
file-level components:

| Licence class | Packages (examples) |
|---|---|
| MIT | annotated-types, anyio, fastapi, filelock, h11, httpx ecosystem, Jinja2 (BSD), markdown-it-py, packaging (Apache/BSD), pydantic, PyYAML, rich, typer |
| Apache-2.0 | gradio, gradio_client, huggingface_hub, openai, opentelemetry-api, python-multipart, tokenizers, tzdata |
| BSD-2/3-Clause | click, fsspec, httpcore, idna, MarkupSafe, networkx, numpy, pandas, Pygments, python-dateutil, starlette, uvicorn |
| PSF-2.0 | typing_extensions, audioop-lts |
| ISC | shellingham |
| MIT-CMU | pillow |
| MPL-2.0 (file-level) | certifi, orjson (also Apache/MIT), tqdm (also MIT) |
| Mixed permissive | torch (Apache/BSD/LLVM exception/BSL), sympy/mpmath (BSD) |

No GPL, AGPL, LGPL, or proprietary package was identified in the installed set.
MPL-2.0 is weak, file-level copyleft: it does not relicense this repository, but
it is recorded here rather than hidden. The corpus is **not** represented as a
redistributable dataset. Weights trained on it are an experimental local
artifact, not a published model.

## 4. Benchmark-contamination incident and correction

The first build (2026-11-19) harvested every tracked `.py` file. That included
`evaluation/native_v01/tasks.py` — the frozen 20-task benchmark itself — and
other evaluation code. A short pretraining run was started on that corpus and
then **stopped before any checkpoint was saved**. The run directory was deleted.

The builder was then changed to be benchmark-safe by construction:

- `evaluation/`, `tests/`, `benchmarks/`, `test/`, `native_v01/`, `native_v02/`,
  `data/`, `runs/`, and build/publication directories are excluded;
- evaluation/benchmark harness filenames (`eval_*`, `bench_*`, `*benchmark*`)
  are excluded;
- after writing, the corpus is scanned for frozen-benchmark markers and the
  build **fails closed** if any marker survives;
- the builder excludes its own source, because it defines the marker strings
  and would otherwise make the scan match itself.

The rebuilt corpus passed that scan. The contaminated run is not used for any
result in this repository. The self-exclusion rule was added after a
reproducibility rebuild caught exactly that failure mode; the corrected build
reproduces the recorded corpus SHA-256.

## 5. What this corpus is not

- It is not The Stack, StarCoderData, or any external corpus.
- It is not benchmark-clean by hope; it is benchmark-clean by an explicit scan.
- It is not a licence-cleared public release.
- It is not enough to produce a competitive coding model. At 28.8M parameters
  and 13.7M tokens on CPU, it is a bounded feasibility experiment.
