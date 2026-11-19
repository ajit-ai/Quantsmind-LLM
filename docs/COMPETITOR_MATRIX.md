# Taksh — Coding AI Competitor Matrix

**Date:** 2026-11-14
**Scope:** coding AI models and coding-agent products relevant to a sovereign, locally deployable coding LLM.
**Evidence policy:** competitor benchmark and architecture figures below are vendor-reported unless marked otherwise. They are directional and must be re-verified before external use. Taksh's own scores are not filled in as competitive numbers because Taksh has no demonstrated coding capability today.

---

## 1. Near-term coding models

| Product | Vendor / source | Parameters | Context | Languages | License | Reported coding result | Taksh gap |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder | Alibaba / Qwen | 0.5B–32B | 32K–128K | 92 | Apache-2.0 (except 3B research terms) | HumanEval instruct: 88.4 (7B), 89.6 (14B), 92.7 (32B) | No model; no 92-language corpus; 2,048 canonical context |
| Qwen3-Coder (verify) | Alibaba / Qwen | 30B-A3B and larger, per Qwen line | reported line-specific | not independently re-verified here | line-specific | Qwen reports SWE-bench-oriented gains for the newer line | Not usable as a hard comparison until re-verified against official cards |
| DeepSeek-Coder-V2 | DeepSeek | 16B / 236B, MoE | 128K | 338 | MIT code + DeepSeek model license | HumanEval 86.4 (Lite) / 94.9 (Full); LiveCodeBench 43.4 | Orders of magnitude more data; Taksh is pre-capability |
| Seed-Coder | ByteDance Seed | 8B | 32K / 64K reasoning | not enumerated publicly | MIT | HumanEval 77.4 base; MBPP 82.0 | Taksh has no 6T-token-scale data or 8B artifact |
| StarCoder2 15B | BigCode / ServiceNow / HF / NVIDIA | 15B | 16K (4K sliding) | 600+ | BigCode OpenRAIL-M | HumanEval 46.3 base / 72.6 instruct | Taksh has one language and no comparable data |
| Codestral | Mistral AI | 22B | 32K (later 128K) | 80+ | MNPL for 22B weights; later terms vary | vendor HumanEval figures; strong FIM | License and product line differ; Taksh has no product weights |
| CodeLlama 34B | Meta | 34B | 16K | ~15 | Llama 2 Community License | HumanEval 48.8 (vendor) | No trained comparable model; licensing differs |

---

## 2. Agent/product competitors

| Product | Positioning | What it does well | Taksh implication |
|---|---|---|---|
| Cursor | AI code editor and agent | Fast iterative editing, cloud models, repo context, IDE distribution | Distribution and UX are the moat; Taksh needs a local/sovereign reason to exist |
| Claude Code | Terminal coding agent | High-quality coding intelligence and tool use | Capability anchor; Taksh must match code evidence, not copy UI |
| GitHub Copilot | IDE-embedded completion | ubiquity, latency, enterprise integrations | Taksh needs measured latency and secure local deployment |
| Cline / Continue | Open IDE agent frameworks | local model option, user control, open workflow | Likely first integration layer for a sovereign coding model |
| Devin / SWE-agent-class products | Autonomous software engineering | multi-file autonomous work | v3.0.0 target only; high evidence and safety bar |
| Sarvam / Krutrim / BharatGPT | India-sovereign general/enterprise models | local/sovereign deployment, India enterprise relationships | Not coding-specialized; Taksh differentiates on coding depth and auditability |

---

## 3. Capability axes comparison

| Axis | Best coding alternatives | Taksh today | Direction |
|---|---|---|---|
| Code generation | Qwen / DeepSeek / Seed-Coder | 0/30 frozen | first real checkpoint before any comparison |
| FIM completion | Codestral / Qwen / StarCoder2 | not implemented | tokenizer v2 with FIM and control tokens |
| Repo context | Claude Code / Cursor / Copilot | engineering only | local repo-context engine after model exists |
| Multi-file engineering | Claude Code / SWE-agent-class tools | planned, not measured | L2.x milestones |
| Agentic repair | SWE-bench-class systems | bounded orchestrator, no model result | v3.0.0 target |
| Licensing | Qwen/Seed-Coder/StarCoder2 permissive lines | Apache-2.0 source; weights unreleased | publication decision remains owner-gated |
| Data sovereignty | local deployments possible, but models may be foreign | core differentiator | keep local, inspectable and auditable |
| Evidence | competitors publish benchmarks | no demonstrated capability | frozen, hashed benchmarks and raw outputs |

---

## 4. Honest white-space

There is a real opening for **sovereign coding AI**, not because incumbents do not exist, but because many regulated teams cannot or will not place source code into a general external service.

Taksh's white-space is:

1. locally deployed coding intelligence;
2. own weights and tokenizer;
3. auditable prompts, traces, checkpoints and data lineage;
4. controlled, deterministic evaluation;
5. explicit safety and permission boundaries;
6. coding-specific depth instead of broad assistant scope;
7. B2B2C / enterprise white-label delivery.

Taksh is not currently competitive with Qwen2.5-Coder, DeepSeek-Coder, Seed-Coder, Codestral or mature IDE agents on capability. It is only meaningfully differentiated if it becomes a **measured sovereign coding model**, not merely a sovereign architecture on paper.

---

## 5. Sources

- Qwen2.5-Coder model cards and technical report: <https://huggingface.co/Qwen/Qwen2.5-Coder-32B>, <https://arxiv.org/abs/2409.12186>
- Qwen/Qwen3-Coder line: <https://www.promptquorum.com/prompt-bites/best-qwen-model-for-coding> (comparison is directional; verify on Hugging Face)
- DeepSeek-Coder-V2: <https://github.com/deepseek-ai/DeepSeek-Coder-V2>, <https://arxiv.org/abs/2406.11931>
- Seed-Coder: <https://github.com/ByteDance-Seed/Seed-Coder>, <https://arxiv.org/abs/2506.03524>
- StarCoder2: <https://huggingface.co/bigcode/starcoder2-15b>, <https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/starcoder2_15b/1.0>
- Codestral: <https://mistral.ai/news/codestral>, <https://huggingface.co/mistralai/Codestral-22B-v0.1>
- CodeLlama: <https://huggingface.co/codellama/CodeLlama-34b-hf>
- Sarvam: <https://docs.sarvam.ai/changelog/2026/8/18>, <https://www.sarvam.ai/epoch/summary>
- Taksh current capability: `docs/CAPABILITY_STATEMENT.md`, `README.md`, `docs/L076_MASTER_ROADMAP_REPORT.md`, frozen Python benchmark records.
