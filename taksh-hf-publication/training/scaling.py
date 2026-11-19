"""Model capacity and compute feasibility estimates (Taksh v1.6.0).

Everything in the scenario table is an **estimate** derived from a stated
formula. The only measured quantities in this repository are the S2 profile
and the throughput figures recorded by the v1.5.0 diagnostic run; they are
labelled as measured and are never mixed into the estimates.

No scenario here is authorised, provisioned or implemented. The purpose is to
make the next scaling decision informed rather than intuitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BYTES_PER_PARAMETER = 4  # float32 weights
OPTIMIZER_STATES = 2  # AdamW first and second moments


@dataclass(frozen=True)
class ModelProfile:
    """A geometry, used for the memory and compute estimates."""

    name: str
    parameters: int
    vocab_size: int
    d_model: int
    layers: int
    context_length: int
    kv_heads: int = 8
    heads: int = 8
    ffn_dim: int = 1024
    tie_embeddings: bool = True
    measured: bool = False
    notes: str = ""

    def memory(self) -> dict:
        """Weight, optimizer and total memory for float32 training."""
        weight_bytes = self.parameters * BYTES_PER_PARAMETER
        optimizer_bytes = self.parameters * BYTES_PER_PARAMETER * OPTIMIZER_STATES
        embedding_rows = self.vocab_size * self.d_model
        return {
            "weight_bytes": weight_bytes,
            "optimizer_bytes": optimizer_bytes,
            "total_bytes": weight_bytes + optimizer_bytes,
            "total_gib": (weight_bytes + optimizer_bytes) / (1024 ** 3),
            "embedding_rows": embedding_rows,
            "embedding_share": (
                embedding_rows / self.parameters if self.parameters else 0.0
            ),
        }

    def activations_note(self) -> str:
        """Activation memory scales with layers x context x batch x d_model."""
        return (
            "activations scale with layers x context x batch x d_model; at "
            f"{self.layers} layers, context {self.context_length} and batch 1, "
            f"a float32 forward pass holds roughly "
            f"{self.layers * self.context_length * self.d_model * 4 / (1024 ** 2):.1f} MiB "
            "of intermediate tensors per sequence (estimate)"
        )

    def training_flops(self, tokens: int) -> dict:
        """Approximate training compute for a token budget.

        The standard dense-transformer estimate is 6 * N * D FLOPs, with an
        attention term that grows with context length. It is an order-of-
        magnitude guide, not a measurement.
        """
        dense = 6 * self.parameters * tokens
        attention = 12 * self.layers * self.d_model * self.context_length * tokens
        return {
            "tokens": tokens,
            "dense_flops": dense,
            "attention_flops": attention,
            "total_flops": dense + attention,
            "total_pflops": (dense + attention) / 1e15,
        }


#: The shipped S2 rung, measured from the repository.
S2 = ModelProfile(
    name="qm-code-m3-9m (S2)",
    parameters=9_359_104,
    vocab_size=5822,
    d_model=256,
    layers=8,
    context_length=512,
    kv_heads=4,
    heads=8,
    ffn_dim=1024,
    tie_embeddings=True,
    measured=True,
    notes="the shipped rung; 9,359,104 parameters, 5,822 embedding rows",
)

#: Planning scenarios. Hypothetical, not authorised, not implemented.
SCENARIOS = (
    ModelProfile(
        name="S2 (current, measured)",
        parameters=9_359_104,
        vocab_size=5822,
        d_model=256,
        layers=8,
        context_length=512,
        kv_heads=4,
        measured=True,
    ),
    ModelProfile(
        name="~30M (code_30m geometry)",
        parameters=16_257_280,
        vocab_size=32_768,
        d_model=256,
        layers=8,
        context_length=2048,
        kv_heads=4,
        notes="the orphan config in configs/model/code_30m.yaml; no matching "
        "tokenizer exists, so it is not servable today",
    ),
    ModelProfile(
        name="~100M",
        parameters=100_000_000,
        vocab_size=32_768,
        d_model=768,
        layers=12,
        context_length=2048,
    ),
    ModelProfile(
        name="~300M",
        parameters=300_000_000,
        vocab_size=32_768,
        d_model=1024,
        layers=24,
        context_length=4096,
    ),
    ModelProfile(
        name="~1B",
        parameters=1_000_000_000,
        vocab_size=32_768,
        d_model=2048,
        layers=24,
        context_length=4096,
    ),
)


def cpu_feasibility(profile: ModelProfile, *, budget_gib: float = 8.0) -> str:
    """Whether the scenario's optimizer state could fit a CPU budget."""
    total = profile.memory()["total_gib"]
    if total <= budget_gib:
        return f"fits a {budget_gib:.0f} GiB CPU budget"
    return f"exceeds a {budget_gib:.0f} GiB CPU budget (needs {total:.1f} GiB)"


def scenario_table(token_budget: int = 100_000_000) -> list:
    """The planning table: memory, compute and CPU feasibility per scenario."""
    rows = []
    for profile in SCENARIOS:
        memory = profile.memory()
        compute = profile.training_flops(token_budget)
        rows.append(
            {
                "name": profile.name,
                "measured": profile.measured,
                "parameters": profile.parameters,
                "vocab_size": profile.vocab_size,
                "d_model": profile.d_model,
                "layers": profile.layers,
                "context_length": profile.context_length,
                "weight_gib": memory["weight_bytes"] / (1024 ** 3),
                "optimizer_gib": memory["optimizer_bytes"] / (1024 ** 3),
                "total_gib": memory["total_gib"],
                "embedding_share": memory["embedding_share"],
                "pflops_for_budget": compute["total_pflops"],
                "activations": profile.activations_note(),
                "cpu_feasibility": cpu_feasibility(profile),
                "notes": profile.notes,
            }
        )
    return rows


def pathways() -> list:
    """Three resource-aware pathways. None is authorised by this module."""
    return [
        {
            "pathway": "1. strictly local CPU-only research",
            "scope": "S2 rung, short authorised runs, syntax-level evaluation",
            "cost": "no spend; hours of wall clock per run",
            "capability": "diagnostics and pipeline evidence only",
            "recommendation": "the correct next step, because no corpus exists",
        },
        {
            "pathway": "2. limited-budget accelerated training",
            "scope": "one authorised run on rented/single-accelerator hardware",
            "cost": "not provisioned, not priced, not authorised here",
            "capability": "a first honest Python-pretraining signal",
            "recommendation": "only after a corpus is approved and a budget is agreed",
        },
        {
            "pathway": "3. larger-scale research with dedicated compute",
            "scope": "multi-node or multi-GPU training of a larger rung",
            "cost": "capital or significant rental spend",
            "capability": "not implied by parameter count; needs its own evidence",
            "recommendation": "premature; defer until a corpus, a tokenizer "
            "decision and a measurable baseline justify it",
        },
    ]
