"""TAKSH Local Coding Assistant (L0.1) -- experimental local model inference.

Launch (from the repository root):

    python -m serving.local_assistant --ckpt runs/v180_sft/checkpoint_last.pt \
        --tokenizer tokenizer/taksh_code_bpe_v1.json

Local-only guarantees:

* binds to loopback hosts only (127.0.0.1 / localhost / ::1); anything else is
  refused at startup;
* no external LLM APIs, no hosted models, no paid services -- there is no
  fallback path of any kind;
* no usage analytics or tracking (Gradio analytics are disabled here);
* no internet access required once dependencies and model files exist;
* CPU inference by default. CUDA is used only if the operator explicitly
  passes ``--device cuda`` on a machine that already has a working build.

The interface shows exactly which checkpoint and tokenizer are loaded (path,
SHA-256, step, parameter count, vocabulary size) and states plainly that this
is an experimental local model, not a reliable coding assistant.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Disable Gradio analytics/usage tracking before gradio is imported anywhere.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.local_inference import generate_ids, load_model  # noqa: E402
from training.chat_template import SYSTEM_PROMPT, render_for  # noqa: E402

#: Loopback hosts are the only hosts this application may bind.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 7861
DEFAULT_CKPT = "runs/v180_sft/checkpoint_last.pt"
DEFAULT_TOKENIZER = "tokenizer/taksh_code_bpe_v1.json"

DEFAULT_MAX_NEW_TOKENS = 64
MAX_NEW_TOKENS_LIMIT = 256
MIN_NEW_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0  # greedy by default: conservative and reproducible

EXPERIMENTAL_BANNER = (
    "**Experimental local model.** This is a small, from-scratch model trained "
    "by the Taksh project itself. It is not a reliable coding assistant, its "
    "answers are frequently wrong, and it is provided for research and "
    "evaluation only. It does not call any external service."
)

SUGGESTED_PROMPT = (
    "Write a Python function named is_even(n) that returns True if n is even "
    "and False otherwise. Include a short example."
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_host(host: str) -> str:
    """Return ``host`` if it is a loopback address, else raise ValueError."""
    normalized = (host or "").strip().lower()
    if normalized not in LOOPBACK_HOSTS:
        raise ValueError(
            f"Refusing to bind to {host!r}. The TAKSH local assistant only serves "
            "loopback addresses (127.0.0.1, localhost, ::1). Exposing the "
            "interface to the network is not supported."
        )
    return normalized


def validate_port(port: int) -> int:
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {port}.")
    return port


def load_runtime(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cpu",
):
    """Load checkpoint + tokenizer with clear, actionable errors.

    Raises RuntimeError with a human-readable message on any failure. Never
    falls back to another model, another format, or an external service.
    """
    ckpt = Path(checkpoint_path)
    tok_path = Path(tokenizer_path)
    if not ckpt.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {ckpt}. Pass --ckpt with the path to a "
            "trusted Taksh .pt checkpoint. The assistant will not substitute "
            "another model."
        )
    if not tok_path.is_file():
        raise RuntimeError(
            f"Tokenizer not found: {tok_path}. Pass --tokenizer with the matching "
            "Taksh BPE tokenizer JSON. The assistant will not substitute another "
            "vocabulary."
        )
    try:
        model, model_cfg = load_model(str(ckpt), device)
    except Exception as exc:  # noqa: BLE001 - present a clean operator-facing error
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt}: {type(exc).__name__}: {exc}. "
            "Check that the file is a trusted Taksh checkpoint trained with the "
            "custom architecture and that it matches the configured model config."
        ) from exc
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load tokenizer {tok_path}: {type(exc).__name__}: {exc}. "
            "Pass the tokenizer that was used to train this checkpoint."
        ) from exc

    vocab = tokenizer.get_vocab_size()
    expected = int(model_cfg.get("vocab_size", -1))
    if vocab != expected:
        raise RuntimeError(
            f"Tokenizer/checkpoint mismatch: tokenizer {tok_path} has vocabulary "
            f"size {vocab}, but the checkpoint was trained with vocab_size "
            f"{expected}. Refusing to run with incompatible artifacts."
        )
    for marker in ("<bos>", "<eos>"):
        if tokenizer.token_to_id(marker) is None:
            raise RuntimeError(
                f"Tokenizer {tok_path} does not register {marker!r} as a single "
                "token; it cannot render the Taksh chat template safely."
            )
    meta = {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "tokenizer_path": str(tok_path),
        "tokenizer_sha256": sha256_file(tok_path),
        "vocabulary_size": vocab,
        "parameters": model.num_params(),
        "d_model": model_cfg.get("d_model"),
        "n_layers": model_cfg.get("n_layers"),
        "max_seq_len": model_cfg.get("max_seq_len"),
        "device": device,
        "torch_version": torch.__version__,
    }
    return model, tokenizer, meta


def build_prompt(tokenizer, message: str) -> str:
    """Render the user message with the same template SFT used (contract from
    the tokenizer itself, so train/serve cannot drift)."""
    return render_for(tokenizer, SYSTEM_PROMPT, message)


def create_demo(model, tokenizer, meta, device: str = "cpu"):
    import gradio as gr

    def model_card() -> str:
        step = meta.get("checkpoint_step_raw")
        lines = [
            "### Loaded model",
            f"- Checkpoint: `{meta['checkpoint_path']}`",
            f"- Checkpoint SHA-256: `{meta['checkpoint_sha256']}`",
        ]
        if step is not None:
            lines.append(f"- Training step: {step}")
        lines += [
            f"- Parameters: {meta['parameters']:,} (d_model={meta['d_model']}, "
            f"layers={meta['n_layers']})",
            f"- Vocabulary: {meta['vocabulary_size']} tokens",
            f"- Max context: {meta['max_seq_len']} tokens",
            f"- Tokenizer: `{meta['tokenizer_path']}`",
            f"- Tokenizer SHA-256: `{meta['tokenizer_sha256']}`",
            f"- Device: {meta['device']} (torch {meta['torch_version']})",
        ]
        return "\n".join(lines)

    def respond(message, history, max_new_tokens, temperature):
        history = list(history or [])
        message = (message or "").strip()
        if not message:
            yield history, "", "Enter a prompt first."
            return
        history.append({"role": "user", "content": message})
        yield history, "", "Generating on CPU… (a 28M-parameter model still takes a moment)"
        prompt = build_prompt(tokenizer, message)
        try:
            result = generate_ids(
                model,
                tokenizer,
                prompt,
                max_new_tokens=min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT),
                temperature=float(temperature),
                top_k=50,
                device=device,
                binding_repair=True,
            )
        except torch.cuda.OutOfMemoryError:
            history.append(
                {"role": "assistant", "content": "Generation failed: CUDA out of memory. "
                 "Use --device cpu or reduce max new tokens."}
            )
            yield history, "", "Out of memory. Generation aborted safely."
            return
        except Exception as exc:  # noqa: BLE001 - never die with a bare traceback
            history.append(
                {"role": "assistant",
                 "content": f"Generation failed: {type(exc).__name__}: {exc}"}
            )
            yield history, "", "Generation failed. See the message above for details."
            return
        text = result["text"].strip() or "(the model produced no tokens)"
        history.append({"role": "assistant", "content": text})
        stats = (
            f"finish_reason={result['finish_reason']} · "
            f"prompt_tokens={result['prompt_tokens']} · "
            f"completion_tokens={result['completion_tokens']}"
        )
        yield history, "", stats

    def clear():
        return [], "", "Conversation cleared."

    with gr.Blocks(title="TAKSH — Local Coding Assistant (Experimental)") as demo:
        gr.Markdown("# TAKSH — Local Coding Assistant")
        gr.Markdown(EXPERIMENTAL_BANNER)
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                message = gr.Textbox(
                    label="Programming instruction",
                    placeholder=SUGGESTED_PROMPT,
                    lines=2,
                )
                with gr.Row():
                    generate = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear conversation")
                status = gr.Markdown("Ready.")
            with gr.Column(scale=2):
                gr.Markdown(model_card())
                max_new_tokens = gr.Slider(
                    minimum=MIN_NEW_TOKENS,
                    maximum=128,
                    step=8,
                    value=DEFAULT_MAX_NEW_TOKENS,
                    label="Max new tokens (CPU-conscious default; cap 128)",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE,
                    label="Temperature (0.0 = greedy/deterministic)",
                )
        inputs = [message, chatbot, max_new_tokens, temperature]
        outputs = [chatbot, message, status]
        generate.click(respond, inputs=inputs, outputs=outputs)
        message.submit(respond, inputs=inputs, outputs=outputs)
        clear_btn.click(clear, inputs=None, outputs=[chatbot, message, status])
    return demo


def _read_checkpoint_step(checkpoint_path: str):
    """Best-effort step read; never raises (metadata only, post-load)."""
    try:
        from serving.local_inference import torch_load_checkpoint

        blob = torch_load_checkpoint(checkpoint_path)
        return blob.get("step")
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="TAKSH Local Coding Assistant (L0.1) — experimental, local-only."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to a trusted Taksh .pt checkpoint")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Matching Taksh tokenizer JSON")
    parser.add_argument("--host", default=SERVER_NAME, help="Loopback host only (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    try:
        host = validate_host(args.host)
        port = validate_port(args.port)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "error: --device cuda requested but CUDA is not available in this "
            "PyTorch build. Use the default CPU path.",
            file=sys.stderr,
        )
        return 2

    try:
        model, tokenizer, meta = load_runtime(args.ckpt, args.tokenizer, args.device)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    meta["checkpoint_step_raw"] = _read_checkpoint_step(args.ckpt)
    if args.max_new_tokens and not MIN_NEW_TOKENS <= args.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        print(
            f"error: --max-new-tokens must be between {MIN_NEW_TOKENS} and "
            f"{MAX_NEW_TOKENS_LIMIT}, got {args.max_new_tokens}.",
            file=sys.stderr,
        )
        return 2

    print(
        f"TAKSH Local Coding Assistant — experimental local model\n"
        f"  checkpoint : {meta['checkpoint_path']}\n"
        f"  sha256     : {meta['checkpoint_sha256']}\n"
        f"  parameters : {meta['parameters']:,}  vocab={meta['vocabulary_size']}  "
        f"device={meta['device']}\n"
        f"  serving    : http://{host}:{port} (loopback only, no external APIs)",
        flush=True,
    )
    demo = create_demo(model, tokenizer, meta, device=args.device)
    demo.launch(server_name=host, server_port=port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
