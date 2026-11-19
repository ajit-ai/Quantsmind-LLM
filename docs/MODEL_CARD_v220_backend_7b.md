# Taksh v2.2.0 — Model Card: qm-code-v220-backend-7b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v220-backend-7b |
| Version | v2.2.0 |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Base model | qm-code-v200-7b |
| Training data | Backend code (REST APIs, databases, auth, DevOps) |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Domain-specific release candidate |

## Intended Use

Backend-specific coding tasks:
- REST/GraphQL API design
- Database schemas and queries
- Authentication & authorization
- DevOps & CI/CD pipelines
- Error handling and logging

## Training Data

~50K instruction→code pairs focused on backend development, including:
- REST API patterns
- Database schema design
- Authentication patterns (JWT, OAuth)
- DevOps configurations
- Error handling patterns

## Evaluation

| Metric | Target |
|---|---|
| Backend task pass@1 | > 75% |
| API design correctness | > 80% |
| Database query correctness | > 85% |
| Authentication security | > 90% |

## Citation

```
@misc{taksh-v220-backend-7b,
  title = {Taksh v2.2.0: Backend Domain Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

Research software. Generated code should be reviewed before execution.
