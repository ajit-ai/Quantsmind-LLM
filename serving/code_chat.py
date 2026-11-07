import os
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.local_inference import generate_ids, load_model
from training.chat_template import SYSTEM_PROMPT, render_for, render_prompt

#: One canonical preamble. There must not be a second "serving system prompt":
#: the M7 mismatch was exactly that. Callers may still pass their own system
#: message explicitly, but the default is the one training used.
CODE_SYSTEM_PROMPT = SYSTEM_PROMPT

LANGUAGE_INFO = {
    "python": {"extensions": [".py"], "prompt_prefix": "# Python code:\n"},
    "javascript": {"extensions": [".js", ".jsx", ".mjs"], "prompt_prefix": "// JavaScript code:\n"},
    "typescript": {"extensions": [".ts", ".tsx"], "prompt_prefix": "// TypeScript code:\n"},
}


@torch.no_grad()
def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, device: str = "cpu") -> str:
    """Generate a completion. Delegates to the shared sampler in local_inference."""
    result = generate_ids(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )
    return result["text"]


def format_prompt(message: str, language: str, context: str = "", tokenizer=None) -> str:
    """Render the inference prompt.

    C3: this used to hand-build "<|system|>...<|user|>...<|assistant|>", none of
    which exist in the tokenizer, so inference fed the model the same shredded
    byte fragments SFT trained on. It now renders through the shared template so
    training and inference cannot drift apart. With a tokenizer supplied, the
    contract is derived from that tokenizer instead of assumed.
    """
    if context:
        lang_label = language if language != "auto" else "unknown"
        user = f"Context ({lang_label}):\n```\n{context}\n```\n\nQuestion: {message}"
    else:
        user = message
    if tokenizer is None:
        return render_prompt(CODE_SYSTEM_PROMPT, user)
    return render_for(tokenizer, CODE_SYSTEM_PROMPT, user)


def create_code_chat_ui(model, tokenizer, device: str):
    def respond(message, history, language, context):
        if not message.strip():
            yield ""
            return
        prompt = format_prompt(message, language, context)
        full = ""
        for chunk in generate_code(model, tokenizer, prompt, device=device):
            full += chunk
            yield full

    with gr.Blocks(title="Taksh Coder", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Taksh Coder — AI Programming Assistant")
        gr.Markdown("Python, JavaScript/TypeScript specialist, and experimental C# evaluation support. Code generation, debugging, review, and explanation.")

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
            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[language, context],
                    examples=[
                        ["Write a function to debounce a callback in JavaScript", "python", ""],
                        ["Explain how Python decorators give with an example", "python", ""],
                        ["Write a TypeScript generic for a key-value store", "typescript", ""],
                        ["Review this Python code for bugs", "python", "def divide(a, b):\n    return a / b"],
                    ],
                )
    return ui


def main():
    import argparse

    p = argparse.ArgumentParser(description="Taksh Coder — Code Chat UI")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model, model_cfg = load_model(args.ckpt, device)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print(f"Starting Taksh Coder on http://localhost:{args.port}")
    ui = create_code_chat_ui(model, tokenizer, device)
    ui.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
