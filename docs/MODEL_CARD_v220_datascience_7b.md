# Taksh v2.2.0 — Model Card: qm-code-v220-datascience-7b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v220-datascience-7b |
| Version | v2.2.0 |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Base model | qm-code-v200-7b |
| Training data | Data science code (Pandas, PyTorch, SQL, visualization) |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Domain-specific release candidate |

## Intended Use

Data science-specific coding tasks:
- Pandas data manipulation
- PyTorch model training
- SQL queries
- Data visualization
- EDA pipelines

## Training Data

~50K instruction→code pairs focused on data science, including:
- Pandas DataFrame operations
- PyTorch model definitions and training loops
- SQL queries and aggregations
- Matplotlib/Seaborn visualizations
- EDA and data cleaning pipelines

## Evaluation

| Metric | Target |
|---|---|
| Data science task pass@1 | > 75% |
| Pandas operation correctness | > 80% |
| SQL query correctness | > 85% |
| PyTorch model correctness | > 70% |

## Citation

```
@misc{taksh-v220-datascience-7b,
  title = {Taksh v2.2.0: Data Science Domain Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

Research software. Generated code should be reviewed before execution.
