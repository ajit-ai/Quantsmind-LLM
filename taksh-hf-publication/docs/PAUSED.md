# PROJECT STATUS: PAUSED (by founder decision, 2026-08-24)

**State:** Fully preserved · **Spend to date: ₹0** · **Recurring costs: NONE** (no hosting, no cloud, no paid services)
**Resume trigger:** founder says "continue" — pick up exactly at Phase 1 → pilot analysis.

---

## Where we stopped (exact position)

```
Phase 0  Foundations          [██████████] 100% ✅
Phase 1  Data + Pilot         [████████░░] 85% 🟡 PAUSED HERE
Phase 2  1.5B Pretrain        [░░░░░░░░░░] 0% (gated on pilot + future hardware decision)
Phase 3  Post-training        [░░░░░░░░░░] 0%
Phase 4  B2B2C Product        [░░░░░░░░░░] 0%
Phase 5  Pilot launch         [░░░░░░░░░░] 0%
Phase 6  7B / commercial      [░░░░░░░░░░] 0% (future: possible commercial path)
```

## What exists and is safe (on GitHub, free forever)

| Asset | Location |
|---|---|
| From-scratch Transformer + training engine | `training/model.py`, `training/pretrain.py` |
| Own tokenizer (49,152 BPE) | `tokenizer/qm_bpe_v0.json` |
| Corpus v0 (9,993,983 tokens, PII-scrubbed, deduped, **NOT decontaminated**) | `data/corpus_v0.*` |
| India-finance eval set (40 items) + harness | `evaluation/` |
| Local chat UI with India-residency guard | `serving/app.py` |
| Kaggle pilot kit (optional, free tier) | `scripts/kaggle_pilot.ipynb` + guide |
| Full design docs + ADRs + competitive analysis | `docs/` |
| 46 passing tests, CI on every push | `.github/workflows/ci.yml` |

## The decided strategy (unchanged, waiting)

**ADR-0003 (FINAL):** from-scratch sovereign LLM. Ladder: 300M-class pilot → 1.5B
"Quantsmind-Bharat" → 7B when funded. Open models = benchmark rivals only.

**Prototype-first refinement (founder, 2026-08-24):** next build step is a LOCAL
prototype (~15–25M params) trained on the founder's own machine (MX150/CPU), zero
rupees, no cloud dependence. Growth later = better hardware, same code.

## Resume playbook (pick up in 10 minutes)

1. `git clone https://github.com/ajit-ai/Quantsmind-LLM.git && cd Quantsmind-LLM`
2. `python -m pip install -r requirements.txt`
3. `python -m pytest tests -q` (should be 46 passed)
4. Say "continue" → next actions:
   - a. If Kaggle pilot (Experiment 001) finished: paste `pilot_report.json` → gate decision
   - b. If going local-prototype: launch L1 restructure + local `qm-small` training
5. Optional future paths when ready: commercial use, investment, hosted product —
   all documented in `docs/ROADMAP.md` Phases 2–6.

## Rules that stay in force on resume
- Every phase: commit to `develop`, merge to `main`
- No spend without an explicit founder decision
- India-first, sovereign weights, open-models-as-rivals-only
