# Taksh — Rule Book (Binding)

> Owner / sole contributor: **ajit-ai**. No other contributor is added.
> Every phase ends with `develop -> main` merge + docs publish + CI/CD green.

## 1. Ownership — solo contributor lock

- Sole owner + maintainer + reviewer: `ajit-ai`.
- Do NOT invite collaborators. Pending invitations must stay empty.
- External input only via Issues (discussion), never direct push.
- `CODEOWNERS` maps every path to `@ajit-ai`.
- PR review rule: only `@ajit-ai` approves. Any other approval is invalid.
- Verify lock:
  ```bash
  gh api repos/ajit-ai/Quantsmind-LLM/collaborators --jq ".[].login"
  # expected: ajit-ai only
  gh api repos/ajit-ai/Quantsmind-LLM/invitations --jq "."
  # expected: []
  ```

## 2. Branching — strict

- `main` = protected, tagged releases only. NEVER commit directly to `main`.
- `develop` = integration branch. All work lands here first.
- Feature branch per task: `feat/<phase>-<topic>` e.g. `feat/m1-3-exif-fix`.
- Flow: `feat/* -> PR -> develop -> release PR -> main`.
- Every phase end = `develop -> main` merge with eval report attached.
- Model tags: `qm-mm-pilot-300m-vN`, `qm-mm-1.5b-vN`, `qm-chat-1.5b-vN` (git SHA + dataset tag + config hash).

## 3. Commit rules

- Conventional commits: `feat(mm): ...`, `fix(mm): ...`, `docs: ...`, `ci: ...`, `chore: ...`.
- One logical change per commit. `git status` must be explainable in one line.
- Before commit:
  ```bash
  python -m pytest tests -q
  python -m training.validate_manifest --manifest data/mm_m1/MANIFEST_v2.json --tokenizer tokenizer/qm_bpe_v0.json --require-complete-provenance
  # second command only when manifest / image pipeline touched
  ```
- NEVER commit: `runs/*`, `*.pt`, `*.bin`, `*.safetensors`, `*.ckpt`, `data/raw/*`, `data/mm_m1/train/*`, `data/mm_m1/val/*`, `data/mm_m1/test/*` (images), `.env`, secrets, `docs/_build/*`.
- RST docs under `docs/source/*.rst` are source of truth for Pages. HTML under `docs/_build` is NEVER committed.

## 4. Docs publish (RST -> HTML -> GitHub Pages)

- Edit ONLY `docs/source/*.rst` + `docs/*.md` data cards.
- CI job `docs` builds strict: `sphinx-build -W -b html docs/source docs/_build/html`.
- Deploy ONLY from `main` via `peaceiris/actions-gh-pages@v4` to `gh-pages` branch.
- URL: `https://ajit-ai.github.io/Quantsmind-LLM/`.
- Local preview:
  ```bash
  pip install -r docs/requirements-docs.txt
  sphinx-build -W -b html docs/source docs/_build/html
  ```
- Broken RST (warning = error, `-W`) fails CI. Fix RST, never force-merge.
- While repo is `private`: Pages site stays private. After repo goes public, Pages becomes public automatically.

## 5. CI/CD — must be green

- Workflow `.github/workflows/ci.yml` on `push: [develop, main]` + `pull_request`:
  1. `pytest tests -q` (183 tests collected, 2026-10-14)
  2. `import serving.app`
  3. `scripts/smoke_pipeline.sh` tiny CPU train
  4. resume-from-checkpoint check
  5. MM hermetic smoke `synth_images --n 8 + train_mm --max-steps 3`
  6. solo-contributor guard (`github.actor == ajit-ai` on push)
- Workflow `.github/workflows/docs.yml` on `push: main` + `pull_request`: build RST strict (`-W`), deploy Pages only on `main`.
- Red CI blocks merge. No bypass except owner emergency with post-mortem note in release PR.

## 6. Private-repo limitation (documented)

- This repo is currently `private` on Free plan: GitHub API returns `403 Upgrade to Pro or make public` for branch-protection endpoints.
- Therefore `main` protection is enforced by discipline + CI status check, not server-side rule.
- When repo goes public (or Pro): enable `Require pull request + Require status checks (CI, docs) + Require CODEOWNERS review + Dismiss stale reviews`.

## 7. Phase gate checklist (copy into every release PR)

- [ ] `pytest` green, smoke + MM smoke green
- [ ] `STATUS.md` + `ROADMAP.md` boxes updated
- [ ] RST updated, `docs` job green, Pages preview OK
- [ ] Manifest strict gate green if data touched
- [ ] `develop -> main` PR opened by `ajit-ai`, approved by `ajit-ai`, merged, tag pushed
- [ ] Collaborators still only `ajit-ai`, invitations `[]`
