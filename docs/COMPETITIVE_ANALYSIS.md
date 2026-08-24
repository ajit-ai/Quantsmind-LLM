# Competitive Analysis — India Finance-LLM Market (as of Aug 2026)

Living document. Re-verify quarterly; sources noted inline. Feeds ROADMAP strategy decisions.

---

## 1. Landscape snapshot

| Player | What they do | Overlap | Moat | Key facts |
|---|---|---|---|---|
| **StockGro "Stoxo"** | Conversational AI research desk for Indian retail ("India's first stock market AI research engine", Sep 2025) | **Closest competitor** — same Tier-1 use case | 35M+ users, 150+ SEBI-registered analysts, brand | stoxo.club; AI + RA hybrid answers |
| **Univest** | AI screening + SEBI RA-validated picks; advisory-first brokerage | High (AI+analyst B2C) | RA registration INH000013776, 1Cr users, one-click execution | ₹99/mo Pro; human analyst owns every pick |
| **INDmoney** | MCP server exposing portfolio/market data to Claude (May 2026) | Partial (portfolio Q&A) | First-mover MCP, portfolio data graph, read-only design | Core LLM is foreign (Claude) |
| **Sarvam AI** | Sovereign Indian LLMs (30B/105B, 22 languages); **pivoted into BFSI enterprise** (Feb 2026) | High on sovereignty axis | IndiaAI backing, enterprise sales, Indic strength | SBI Life: 3.5 lakh agents, WhatsApp, 11 languages |
| **Krutrim** | Pivoted models → AI cloud services (May 2026) | Low-moderate (infra) | Ola capital, GPU capacity | 25+ enterprise FS/telecom customers; first profit FY26 |
| **ChatGPT / Gemini / Perplexity** | What Indian retail actually uses for finance Q&A today | Behavioral competitor | Capability, habit, price | Zero India compliance/residency guarantees |
| Moneycontrol / Tickertape / Market MOJO | Content + scoring platforms, adding AI features | Low-moderate | Content libraries, user base | Not conversational-first yet |

## 2. White-space map

```
                    India-hosted / sovereign
                            ▲
        Sarvam ●            │            ● Quantsmind
        (general BFSI)      │        (finance-tuned, OWNED weights,
                            │         B2B2C white-label, tool-verified math)
  ChatGPT ●                 │
  (capability, zero         │         ● Stoxo/Univest
   India compliance)        │         (AI + RA, consumer brands,
                            │          foreign/GPT-wrapped cores)
  ──────────────────────────┼──────────────────────────▶
                    Finance-domain depth
```

**Unclaimed quadrant**: finance-domain depth × sovereign/owned weights × B2B2C white-label.
No Indian player combines: owned/adapted model weights + residency-by-default +
white-label B2B2C distribution + tool-grounded arithmetic.

## 3. Regulatory tailwind (validates our architecture)

- **SEBI (Aug 2026, FICCI address)**: forthcoming **tiered AI/ML guidelines** —
  accountability, data controls, human oversight, **kill switches**; regulated entities
  remain responsible for vendor AI outputs ("AI responsibility cannot be outsourced").
- **SEBI RA framework**: an algorithm cannot hold an RA registration; a named,
  NISM-certified analyst must own every published pick → advisory features require a
  **RA partnership**, full stop.
- **Baseline compliance stack for BFSI AI** (RBI IT-Governance MD, SEBI CSCRF, IRDAI
  sandbox, DPDP Act): prompt logging, audit trails, explainability, India data residency,
  model inventory with owners.

**Quantsmind architecture already ships**: immutable per-message audit logs, advisory
refusal boundaries + SEBI-RIA disclaimers, residency guard (ADR-0002), promotion gate
with eval reports, per-tenant kill switches (planned Phase 3). Competitors wrapping
foreign APIs must retrofit all of this.

## 4. Honest gaps (what they have, we don't)

| Gap | Implication | Mitigation |
|---|---|---|
| Users/distribution (Stoxo 35M, Univest 1Cr; we have 0) | No B2C land-and-expand | B2B2C: partners bring users |
| SEBI RA registration | Cannot publish picks/advice ourselves | Partner with RA firm; we power, they own accountability |
| Content/data pipelines (MC/ET-class) | Weaker news/research corpus | License + filings-first strategy; synthetic augmentation |
| Brand trust at retail | End-users won't know us | We're invisible infra — partner's brand, our engine |

## 5. Strategic verdict

**Do not build a Stoxo competitor. Sell to everyone who must respond to Stoxo.**

1. **Sell the compliant engine, not the app** — every broker/wealth platform needs its own
   branded AI answer to Stoxo; SEBI's vendor-accountability stance makes foreign-API
   wrappers a liability → our residency + audit stack is the pitch.
2. **RA partnership model** (Univest pattern): AI screens/explains, RA firm owns
   accountability for anything advisory-shaped.
3. **Window: ~2 quarters.** Sarvam is circling BFSI with sovereign positioning and
   enterprise sales muscle. Our claim to "finance-tuned sovereign engine" must be
   demonstrable (M2) before they verticalize.

## 6. Review cadence

Re-run web scan quarterly; update this file + ROADMAP if any player: (a) launches
white-label/B2B2C engine, (b) ships India-hosted finance-tuned model, (c) SEBI finalizes
AI guidelines (then map our controls to the final text in `safety/`).
