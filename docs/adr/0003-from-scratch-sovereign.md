# ADR-0003: From-scratch sovereign pretrain — FINAL product core

Date: 2026-08-24 · Status: Accepted (founder decision) · Supersedes interim ADR-000 open-base plan

## Context
Two paths existed for the Quantsmind product core:
(A) extend an Apache-2.0 open base (cheaper, stronger raw capability, but "adapted someone else's model");
(B) pretrain from scratch (costlier, but 100% owned weights and an uncopyable "born-in-India,
owned-weights" story). Interim ADR-000 chose (A) on cost grounds.

## Decision
**(B) From-scratch, sovereign.** Ladder: 300M pilot (12B tokens, <₹10K) → 1.5B
"Quantsmind-Bharat" (60B tokens, ₹3–5 lakh) → 7B flagship when funded (₹30–60 lakh).
Open models appear in the eval harness as **benchmark rivals only — never the product core.**

## Rationale
1. Founder mission: weight ownership and sovereignty are the point of the project.
2. Market: no Indian player combines finance-tuned + owned-weights + B2B2C white-label
   (COMPETITIVE_ANALYSIS.md white-space map). Sarvam is general-purpose; Stoxo/Univest
   wrap foreign or third-party cores.
3. Regulatory story: "weights born and trained under our control, zero foreign base"
   is the strongest possible answer to SEBI's vendor-accountability direction.
4. Tier-1 use cases (RAG + tools) do not require GPT-4-class open-ended knowledge;
   domain focus + tool-grounded arithmetic + India corpus close the gap where it pays.
5. All required code already exists in-repo (Phase 0): transformer, tokenizer, pretrain
   loop, eval gate — the from-scratch path was the original design.

## Consequences
- Compute budget becomes a first-class roadmap item (rental GPU provisioning tasks).
- Quality floor is unknown until the pilot runs — mitigated by scaling-law probes and
  the 300M pilot gate before any large spend.
- Public scorecard commitment: "sovereign 1.5B vs open 7B-class on India-finance" is
  reported honestly every release.
- Interim ADR-000 text retained in git history as the documented alternative.
