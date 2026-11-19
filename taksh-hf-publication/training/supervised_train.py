"""Aligned supervised training and evaluation (Taksh v2.1.0).

The training loop consumes :class:`training.supervised.SupervisedExample`
objects: the *whole* text (instruction + signature + body) is presented to the
model, and the loss is masked to the response only. Evaluation uses exactly
the prompt prefix the model was trained on, so training and inference agree.

Metrics are static: the generated program is parsed and compared, never
executed.
"""

from __future__ import annotations

import ast
import math
import time

import torch

from training.determinism import set_determinism
from training.objective import IGNORE_INDEX, accumulation_weights, token_mean_loss
from training.schedules import WarmupCosineSchedule
from training.sequence import Batch


def build_batches(tokenizer, examples, *, max_length: int = 256, eos_token: str = "<eos>", identifier_config=None):
    """Tokenise examples into masked training batches.

    With ``identifier_config=None`` (default) batches carry no weights and
    the objective is exactly the previous token-mean objective. With an
    :class:`~training.identifier_supervision.IdentifierSupervisionConfig`,
    each batch carries a mean-1.0 ``weights`` tensor aligned with its
    labels, computed once per example (never re-applied).
    """
    from training.identifier_supervision import classify_identifiers, loss_weights
    from training.supervised import supervise

    batches = []
    for example in examples:
        supervision = supervise(
            tokenizer, example, max_length=max_length, eos_token=eos_token,
            required_markers=example.metadata.get("markers", ()),
        )
        if not supervision["supervised_positions"]:
            continue
        ids = supervision["ids"]
        labels = supervision["labels"]
        inputs = torch.tensor([ids[:-1]], dtype=torch.long)
        target = torch.tensor(labels, dtype=torch.long)
        weights = None
        if identifier_config is not None:
            occurrences = classify_identifiers(example.signature, example.body)
            weights = torch.tensor(
                loss_weights(supervision, occurrences, tokenizer=tokenizer, config=identifier_config)["weights"],
                dtype=torch.float32,
            )
        batches.append(
            {
                "input_ids": inputs,
                "labels": target,
                "weights": weights,
                "supervised": len(supervision["supervised_positions"]),
                "example_id": example.example_id,
                "operation": example.operation,
            }
        )
    return batches


def weighted_mean_loss(logits, labels, item_weights):
    """Mean loss for one batch, weighted if ``item_weights`` is present.

    Returns ``(micro_mean, supervised_count)`` matching the existing
    ``summed / supervised`` contract of :func:`token_mean_loss`.
    """
    if item_weights is None:
        summed, supervised = token_mean_loss(logits, labels)
        if supervised == 0:
            return None, 0
        return summed / supervised, supervised
    from training.identifier_supervision import weighted_token_mean_loss

    return weighted_token_mean_loss(logits, labels, item_weights)


def _batch_tensor(items) -> Batch:
    inputs = torch.cat([item["input_ids"] for item in items], dim=0)
    labels = torch.cat([item["labels"] for item in items], dim=0)
    attention = torch.ones_like(inputs)
    return Batch(
        input_ids=inputs,
        labels=labels,
        attention_mask=attention,
        supervised_tokens=int(sum(item["supervised"] for item in items)),
        document_count=len(items),
    )


def train_supervised(
    *,
    model_config: dict,
    tokenizer,
    examples,
    val_examples,
    lr: float,
    steps: int,
    max_length: int = 256,
    grad_accum: int = 2,
    seed: int = 20261021,
    identifier_config=None,
    warmup: int = 2,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    artifacts=None,
    device: str = "cpu",
):
    """Train on instruction-aligned examples and return metrics plus the model."""
    from training.model import QMModel

    set_determinism(seed)
    model = QMModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    schedule = WarmupCosineSchedule(
        peak_lr=lr, warmup_steps=warmup, total_steps=steps, min_ratio=0.1
    )
    train_batches = build_batches(tokenizer, examples, max_length=max_length, identifier_config=identifier_config)
    val_batches = build_batches(tokenizer, val_examples, max_length=max_length, identifier_config=identifier_config)
    if not train_batches:
        raise ValueError("no trainable examples were produced")

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(train_batches), generator=generator).tolist()

    history = []
    consumed = 0
    started = time.time()
    for step in range(steps):
        window = []
        for micro in range(grad_accum):
            index = order[(step * grad_accum + micro) % len(order)]
            window.append(train_batches[index])
        counts = [item["supervised"] for item in window]
        weights = accumulation_weights(counts)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for item, weight in zip(window, weights):
            batch = _batch_tensor([item])
            logits, _ = model(batch.input_ids.to(device))
            micro_mean, supervised = weighted_mean_loss(logits, batch.labels.to(device), item.get("weights"))
            if supervised == 0 or micro_mean is None:
                continue
            (micro_mean * weight).backward()
            loss_value += float(micro_mean.item()) * weight
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )
        )
        learning_rate = schedule.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        consumed += sum(counts)
        history.append(
            {
                "step": step + 1,
                "loss": loss_value,
                "lr": learning_rate,
                "grad_norm_before_clip": grad_norm,
                "supervised_tokens": sum(counts),
                "finite_loss": math.isfinite(loss_value),
            }
        )
        if not math.isfinite(loss_value):
            break

    val_loss = _validation_loss(model, val_batches, device)
    result = {
        "lr": lr,
        "steps": len(history),
        "requested_steps": steps,
        "tokens_consumed": consumed,
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "validation_loss": val_loss,
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "train_examples": len(examples),
        "val_examples": len(val_examples),
        "seed": seed,
    }
    if artifacts is not None:
        artifacts.write_metric("training.json", result)
    return result, model


def _validation_loss(model, batches, device) -> float | None:
    if not batches:
        return None
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for item in batches:
            batch = _batch_tensor([item])
            logits, _ = model(batch.input_ids.to(device))
            mean_loss, supervised = weighted_mean_loss(logits, batch.labels.to(device), item.get("weights"))
            if supervised:
                total += float((mean_loss * supervised).item())
            count += supervised
    model.train()
    return (total / count) if count else None


class EvaluationError(ValueError):
    """An evaluation record cannot be assembled safely."""


def reconstruct_program(signature: str, body: str) -> str:
    """Assemble Python source from a signature and a body.

    The natural-language instruction is deliberately excluded: only the
    signature and the generated body are Python source. The same rule is used
    for the reference body and for a generated continuation, so the two are
    compared symmetrically. Indentation carried by the body is preserved; a
    body emitted flush-left is indented once so that it can be parsed as the
    function block.
    """
    if not signature or not signature.strip():
        raise EvaluationError("a signature is required to reconstruct a program")
    signature = signature.rstrip()
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return signature + "\n"
    if lines[0] and not lines[0][0].isspace():
        lines = [f"    {line}" if line else line for line in lines]
    return signature + "\n" + "\n".join(lines) + "\n"


def _expression_operations(node) -> set:
    """Operation tags implied by the value an expression computes."""
    operations = set()
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            operations.add("add")
        elif isinstance(node.op, ast.Sub):
            operations.add("subtract")
        elif isinstance(node.op, ast.Mult):
            operations.add("multiply")
    elif isinstance(node, ast.ListComp):
        for generator in node.generators:
            for condition in generator.ifs:
                for sub in ast.walk(condition):
                    if not isinstance(sub, ast.Compare):
                        continue
                    for op in sub.ops:
                        if isinstance(op, ast.Gt):
                            operations.add("keep_positive")
                        elif isinstance(op, ast.Lt):
                            operations.add("keep_negative")
    elif isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr == "upper":
                operations.add("upper")
            elif function.attr == "lower":
                operations.add("lower")
            elif function.attr == "get":
                operations.add("mapping_get")
        elif isinstance(function, ast.Name) and function.id == "len":
            operations.add("mapping_len")
    return operations


def _selection_operations(statements) -> set:
    """Operations performed by an if/return selection, e.g. a maximum.

    Credit requires a genuine selection: the branch returns one of the two
    values being compared. Merely comparing two values and returning a boolean
    is not a maximum, and an operator in an unrelated expression earns nothing.
    """
    operations = set()
    for statement in statements:
        if not isinstance(statement, ast.If):
            continue
        test = statement.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        direction = None
        if isinstance(test.ops[0], ast.Gt):
            direction = "max_gt"
        elif isinstance(test.ops[0], ast.Lt):
            direction = "min_lt"
        if direction is None:
            continue
        operands = [test.left] + list(test.comparators)
        returned = [
            node.value
            for branch in (statement.body, statement.orelse)
            for node in branch
            if isinstance(node, ast.Return) and node.value is not None
        ]
        if not returned:
            continue
        if all(
            any(ast.dump(value) == ast.dump(operand) for operand in operands)
            for value in returned
        ):
            operations.add(direction)
    return operations


def analyse_program(source: str) -> dict:
    """Static analysis of one reconstructed program. Never executes code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {"parses": False, "operations": set(), "body_dump": None}
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    if not functions:
        return {"parses": False, "operations": set(), "body_dump": None}
    function = functions[-1]
    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    returned_names = {node.id for node in returns if isinstance(node, ast.Name)}
    operations = set()
    for value in returns:
        operations |= _expression_operations(value)
    for node in ast.walk(function):
        if not isinstance(node, ast.AugAssign):
            continue
        target = getattr(node.target, "id", "")
        if not isinstance(node.op, ast.Add) or target not in returned_names:
            continue
        if "count" in target:
            operations.add("count_items")
        elif "total" in target:
            operations.add("sum_items")
    operations |= _selection_operations(function.body)
    return {
        "parses": True,
        "operations": operations,
        "body_dump": ast.dump(
            ast.Module(body=function.body, type_ignores=[]), annotate_fields=False
        ),
    }


def evaluate_supervised(model, tokenizer, examples, *, budget: int = 96,
                        device: str = "cpu") -> dict:
    """Evaluate instruction-to-operation behaviour on a partition.

    The model is prompted with exactly the training prompt prefix, and the
    continuation is parsed and compared statically. No program is executed.
    """
    from serving.local_inference import generate_ids

    if isinstance(examples, dict):
        flat = []
        for split in sorted(examples):
            flat.extend(examples[split])
        examples = flat

    records = []
    for example in examples:
        result = generate_ids(
            model, tokenizer, example.prompt_text, max_new_tokens=budget,
            temperature=0.0, device=device,
        )
        continuation = result.get("text", "")
        markers = example.metadata.get("markers", [])
        reference_source = reconstruct_program(example.signature, example.body)
        generated_source = reconstruct_program(example.signature, continuation)
        reference = analyse_program(reference_source)
        generated = analyse_program(generated_source)
        expected = set(reference["operations"])
        structural = bool(
            generated["parses"] and expected
            and expected.issubset(generated["operations"])
        )
        exact = bool(
            generated["parses"] and reference["parses"]
            and generated["body_dump"] == reference["body_dump"]
        )
        # The historical operator-token metric is retained unchanged so that
        # earlier reports remain comparable. It is a *token* check, not a
        # structural check, and is never used for the structural verdict.
        operator_token = bool(markers) and any(
            marker in (example.prompt_text + continuation) for marker in markers
        )
        records.append(
            {
                "example_id": example.example_id,
                "operation": example.operation,
                "family": example.family,
                "split": example.split,
                "syntax_valid": generated["parses"],
                "complete_function": generated["parses"],
                "parses": generated["parses"],
                "operation_agreement": operator_token,
                "structural_operation": structural,
                "expected_operations": sorted(expected),
                "generated_operations": sorted(generated["operations"]),
                "exact_target": exact,
                "ast_equal": exact,
                "status": (
                    "complete_and_exact" if exact else
                    "complete_correct_operation" if structural else
                    "complete_wrong_operation" if generated["parses"] else
                    "incomplete"
                ),
                "tokens": len(result.get("token_ids") or []),
                "finish_reason": result.get("finish_reason"),
                "eos_terminated": result.get("finish_reason") == "stop",
                "truncated": result.get("finish_reason") == "length",
                "degenerate": _degenerate(continuation, result.get("token_ids") or []),
                "text": continuation,
            }
        )
    return {"records": records, "metrics": supervised_metrics(records)}


def _degenerate(text: str, tokens) -> bool:
    if not tokens:
        return True
    longest = 1
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    if longest >= 8:
        return True
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) >= 2 and len(set(lines)) / len(lines) < 0.5:
        return True
    return False


def supervised_metrics(records) -> dict:
    total = len(records)
    if not total:
        return {"cases": 0}
    return {
        "cases": total,
        "syntax_valid": sum(
            1 for r in records if r.get("syntax_valid", r.get("parses"))
        ),
        "complete_function": sum(
            1 for r in records if r.get("complete_function", r.get("parses"))
        ),
        "operation_agreement": sum(
            1 for r in records if r.get("operation_agreement")
        ),
        "structural_operation": sum(
            1 for r in records if r.get("structural_operation")
        ),
        "exact_target": sum(1 for r in records if r.get("exact_target")),
        "complete_and_exact": sum(
            1 for r in records if r.get("status") == "complete_and_exact"
        ),
        "complete_correct_operation": sum(
            1 for r in records if r.get("status") == "complete_correct_operation"
        ),
        "complete_wrong_operation": sum(
            1 for r in records if r.get("status") == "complete_wrong_operation"
        ),
        "incomplete": sum(1 for r in records if r.get("status") == "incomplete"),
        "degenerate": sum(1 for r in records if r.get("degenerate")),
        "eos_terminated": sum(1 for r in records if r.get("eos_terminated")),
        "truncated": sum(1 for r in records if r.get("truncated")),
        "mean_tokens": round(
            sum(r.get("tokens") or 0 for r in records) / total, 2
        ),
    }
