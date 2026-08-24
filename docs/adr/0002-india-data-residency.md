# ADR-0002: India data residency and regulatory posture

Date: 2026-08-24 · Status: Accepted

## Context
Quantsmind targets Indian financial services (B2B2C). Indian financial regulation and
customer expectations demand that user prompts (which may contain holdings, KYC-adjacent
context) remain under Indian/owner-controlled infrastructure. Free routing to US cloud
LLM APIs is unacceptable as a default.

## Decision
1. **Default-deny egress**: `QM_STRICT_RESIDENCY=1` (default ON). The serving layer
   refuses to send prompts to any endpoint that is not localhost/private-network
   (RFC1918, loopback, .local/.internal hosts). Override requires explicitly setting
   `QM_STRICT_RESIDENCY=0` — a conscious, auditable operator choice.
2. **India-specific system prompt**: SEBI/RBI context, Indian conventions (Rs., lakh/crore,
   NSE/BSE), Indian instruments in examples, and a hard boundary: the assistant is not a
   SEBI-registered investment adviser (RIA) and must not give personalized advice; users
   are directed to SEBI-registered advisers.
3. **Disclaimer layer** names the SEBI RIA boundary explicitly on every response.
4. Production deployment target (Phase 3): Indian-region hosting (e.g., Mumbai regions of
   Indian cloud or Indian DCs of global providers), private VPC endpoints only.

## Consequences
- Partners get a defensible data-residency story for RBI/SEBI-facing compliance.
- Demo/testing against public APIs requires conscious override (documented in README).
- Phase 1 eval set (India-finance held-out) becomes the deciding quality metric.
- Future: allowlist file for partner-approved India-hosted endpoints (`serving/residency_allowlist.yaml`).
