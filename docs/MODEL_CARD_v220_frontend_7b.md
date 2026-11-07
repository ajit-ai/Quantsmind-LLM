# Taksh v2.2.0 — Model Card: qm-code-v220-frontend-7b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v220-frontend-7b |
| Version | v2.2.0 |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Base model | qm-code-v200-7b |
| Training data | Frontend code (React, TypeScript, CSS, HTML, UI components) |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Domain-specific release candidate |

## Intended Use

Frontend-specific coding tasks:
- React component generation
- TypeScript type definitions
- CSS styling and layout
- UI state management
- Accessibility compliance
- UI testing

## Training Data

~50K instruction→code pairs focused on frontend development, including:
- React component patterns
- TypeScript interfaces and types
- CSS frameworks (Tailwind, Material-UI, Bootstrap)
- Accessibility patterns (ARIA, keyboard navigation)
- State management (Redux, Zustand, Context API)

## Evaluation

| Metric | Target |
|---|---|
| Frontend task pass@1 | > 80% |
| Component generation correctness | > 75% |
| Accessibility compliance | > 90% |
| TypeScript type safety | > 85% |

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (training) | 32× A100 | 64× H100 |
| GPUs (inference) | 1× A100 | 2× A100 |

## Citation

```
@misc{taksh-v220-frontend-7b,
  title = {Taksh v2.2.0: Frontend Domain Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

Research software. Generated code should be reviewed before execution.
