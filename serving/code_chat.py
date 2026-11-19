"""Gradio chat UI wired to the TAKSH provider architecture.

This UI follows the provider architecture instead of hardcoding a single
checkpoint path. Every response shows which provider answered and which model
it used. External providers (e.g. Seed-Coder) remain optional, labelled
adapters; native TAKSH is the default and never silently substituted.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
import serving.native_provider  # noqa: F401,E402  (registers the native-taksh provider)

from serving.providers import (  # noqa: E402
    GenerationConfig,
    GenerationRequest,
    ProviderError,
    generate_with_timeout,
    get_provider,
    list_providers,
)
from serving.providers import ProviderNotFoundError  # noqa: E402
from training.chat_template import SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT  # noqa: E402

CODE_SYSTEM_PROMPT = TRAINING_SYSTEM_PROMPT  # exported for compatibility
SYSTEM_PROMPT = TRAINING_SYSTEM_PROMPT  # exported for compatibility

LANGUAGE_INFO = {
    "python": {"extensions": [".py"], "prompt_prefix": "# Python code:\n"},
    "javascript": {"extensions": [".js", ".jsx", ".mjs"], "prompt_prefix": "// JavaScript code:\n"},
    "typescript": {"extensions": [".ts", ".tsx"], "prompt_prefix": "// TypeScript code:\n"},
}


def format_prompt(message: str, language: str, context: str = "", tokenizer=None) -> str:
    """Render the inference prompt through the canonical renderer."""
    if context:
        lang_label = language if language != "auto" else "unknown"
        user = f"Context ({lang_label}):\n```\n{context}\n```\n\nQuestion: {message}"
    else:
        user = message
    if tokenizer is None:
        from training.chat_template import render_prompt  # noqa: E402

        return render_prompt(TRAINING_SYSTEM_PROMPT, user)
    from training.chat_template import render_for  # noqa: E402

    return render_for(tokenizer, TRAINING_SYSTEM_PROMPT, user)


def render_note() -> str:
    return (
        "**TAKSH — coding assistant**\n\n"
        "Native TAKSH (default) and optional external adapters. "
        "Native capability: **_NOT_EVALUATED** (v0.1 baseline: 0/20). "
        "No external provider is silently substituted; the provider that answered "
        "is shown on every response."
    )


def build_provider_card(provider_name: str, provider) -> str:
    """Markdown identity block; never raises."""
    try:
        identity = provider.identity().as_dict()
        caps = provider.capabilities().as_dict()
        lines = [
            f"### Provider: `{identity['provider']}`",
            f"- Model: `{identity['model_id']}`",
            f"- Revision: `{identity['revision']}`",
            f"- Source: `{identity['source']}`",
            f"- Tokenizer: `{identity.get('tokenizer')}`",
            f"- License/rights: {identity['license_or_rights']}",
            f"- Status: `{caps['status']}`",
            f"- Available: `{caps['available']}`",
            f"- Configured: `{caps['configured']}`",
            f"- Offline: `{caps['offline']}`",
            f"- Context: `{caps.get('max_context_tokens')}`",
            f"- Notes: {caps['notes']}",
        ]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - UI must never crash on diagnostics
        return (
            f"### Provider: `{provider_name}`\n\n"
            f"Provider selection failed to build an identity card: "
            f"{type(exc).__name__}: {exc}"
        )


def _load_tokenizer(name_or_path: str):
    from tokenizers import Tokenizer

    candidates = [
        name_or_path,
        f"tokenizer/{name_or_path}" if not name_or_path.startswith("/") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.endswith(".json") and Path(candidate).is_file():
            return Tokenizer.from_file(candidate)
    return None


def make_provider(name: str, checkpoint: str | None, tokenizer_name: str | None, device: str):
    """Factory for the current Gradio session; one of two kinds.

    External providers ignore the checkpoint/tokenizer arguments; the native provider requires them.
    """
    if name == "native-taksh":
        return get_provider(
            "native-taksh",
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer_name,
            device=device,
        )
    return get_provider(name)


def generate_answer(provider, message: str, language: str, context: str,
                    temperature: float, max_new_tokens: int, seed: int) -> str:
    """Generate a response via the provider; fail with an honest message if unavailable."""
    tokenizer = _load_tokenizer(os.environ.get("TAKSH_NATIVE_TOKENIZER") or "taksh_code_bpe_v1.json")
    if tokenizer is None:
        prompt = format_prompt(message, language, context, tokenizer=None)
    else:
        prompt = format_prompt(message, language, context, tokenizer=tokenizer)
    try:
        request = GenerationRequest(
            prompt=prompt,
            config=GenerationConfig(
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=1,
                top_p=1.0,
                seed=int(seed),
                timeout_seconds=120.0,
            ),
        )
        result = generate_with_timeout(provider, request)
        header = (
            f"**Provider:** `{result.provider}` — `{result.model_id}` "
            f"— finish `{result.finish_reason}`\n\n"
        )
        return header + str(result.text)
    except ProviderError as exc:
        return (
            f"**Provider `{provider.name}` failed:** `{type(exc).__name__}: {exc}`\n\n"
            "No fallback is provided. This response is the honest provider error path."
        )
    except Exception as exc:  # noqa: BLE001 - UI must degrade, never fake
        return (
            f"**Provider `{provider.name}` error:** `{type(exc).__name__}: {exc}`\n\n"
            "No fallback is provided. This is the honest error path."
        )


def create_code_chat_ui(checkpoint: str | None, tokenizer_path: str | None, device: str, initial_provider: str):
    device_t = device
    tokenizer_name = tokenizer_path
    initial_name = initial_provider

    def bootstrap(name):
        provider = make_provider(name, checkpoint, tokenizer_name, device_t)
        return provider, build_provider_card(name, provider)

    try:
        provider, card = bootstrap(initial_name)
    except ProviderNotFoundError:
        provider = make_provider("native-taksh", checkpoint, tokenizer_name, device_t)
        card = build_provider_card("native-taksh", provider)
    except Exception as exc:  # noqa: BLE001
        provider = None
        card = (
            f"### Provider setup failed\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Set `TAKSH_NATIVE_CHECKPOINT` / `TAKSH_NATIVE_TOKENIZER` or pass `--ckpt`."
        )

    def apply_selection(name):
        try:
            provider_ = make_provider(name, checkpoint, tokenizer_name, device_t)
            return provider_, build_provider_card(name, provider_)
        except Exception as exc:  # noqa: BLE001
            return None, (
                f"### Provider `{name}` failed to initialize\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "No fallback is provided."
            )

    def respond(message, history, provider_name, language, context, temperature, max_new_tokens, seed):
        if not message.strip():
            return ""
        try:
            provider_ = make_provider(provider_name, checkpoint, tokenizer_name, device_t)
        except Exception as exc:  # noqa: BLE001
            return (
                f"**Provider `{provider_name}` could not be initialized:** "
                f"`{type(exc).__name__}: {exc}`"
            )
        return generate_answer(provider_, message, language, context, temperature, max_new_tokens, seed)

    with gr.Blocks(title="Taksh Coder", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Taksh Coder — coding assistant (provider-wired)")
        gr.Markdown(render_note())
        gr.Markdown(card)

        provider_name = gr.Dropdown(
            choices=list_providers(),
            value=initial_name,
            label="Provider",
        )
        apply_button = gr.Button("Apply provider")
        identity_box = gr.Markdown(card)
        apply_button.click(fn=apply_selection, inputs=[provider_name], outputs=[identity_box])

        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=["auto", "python", "javascript", "typescript", "csharp"],
                    value="auto",
                    label="Language",
                )
                context = gr.Textbox(
                    lines=5,
                    label="Code Context (optional)",
                    placeholder="Paste relevant code here for context...",
                )
                temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Temperature")
                max_new_tokens = gr.Slider(8, 256, value=64, step=8, label="Max new tokens")
                seed = gr.Number(value=1, label="Seed")
            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[provider_name, language, context, temperature, max_new_tokens, seed],
                    examples=[
                        ["Write a function to debounce a callback in JavaScript", "native-taksh", "python", "", 0.0, 64, 1],
                        ["Explain how Python decorators work with an example", "native-taksh", "python", "", 0.0, 64, 1],
                        ["Write a TypeScript generic for a key-value store", "native-taksh", "typescript", "", 0.0, 64, 1],
                        ["Review this Python code for bugs", "native-taksh", "python", "def divide(a, b):\n    return a / b", 0.0, 64, 1],
                    ],
                )
        gr.Markdown(
            "Provider shown on every response. No external provider is silently substituted. "
            "Native TAKSH is the default; external adapters are optional and labelled."
        )
    return ui


def main():
    import argparse

    p = argparse.ArgumentParser(description="Taksh Coder — Code Chat UI (provider-wired)")
    p.add_argument("--ckpt", default=os.environ.get("TAKSH_NATIVE_CHECKPOINT") or "runs/v180_sft/checkpoint_last.pt",
                   help="Path to a qualified checkpoint (native-taksh). Defaults to $TAKSH_NATIVE_CHECKPOINT or v180_sft.")
    p.add_argument("--tokenizer", default=os.environ.get("TAKSH_NATIVE_TOKENIZER") or "tokenizer/taksh_code_bpe_v1.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--provider", default="native-taksh", choices=list_providers())
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = args.device
    print(f"device={device} provider={args.provider}")

    ui = create_code_chat_ui(args.ckpt, args.tokenizer, device, args.provider)
    ui.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
