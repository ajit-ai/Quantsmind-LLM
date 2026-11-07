# Taksh v2.2.0 — Model Card: qm-code-v220-embedded-7b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v220-embedded-7b |
| Version | v2.2.0 |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Base model | qm-code-v200-7b |
| Training data | Embedded/IoT code (C/C++, microcontrollers, RTOS) |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Domain-specific release candidate |

## Intended Use

Embedded/IoT-specific coding tasks:
- C/C++ microcontroller code
- RTOS task scheduling
- Memory-constrained optimization
- Hardware abstraction layers
- Driver development

## Training Data

~50K instruction→code pairs focused on embedded development, including:
- C/C++ driver code
- RTOS patterns (FreeRTOS, task scheduling)
- Memory management
- Hardware abstraction
- Peripheral drivers

## Evaluation

| Metric | Target |
|---|---|
| Embedded task pass@1 | > 70% |
| C/C++ compilation success | > 80% |
| Memory safety | > 75% |
| RTOS correctness | > 70% |

## Citation

```
@misc{taksh-v220-embedded-7b,
  title = {Taksh v2.2.0: Embedded Domain Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

Research software. Generated code should be reviewed before execution.
