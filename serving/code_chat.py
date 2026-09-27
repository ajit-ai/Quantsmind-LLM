import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.local_inference import load_model

CODE_SYSTEM_PROMPT = (
    "You are Quantsmind Coder, an expert programming assistant specializing in "
    "Python and JavaScript/TypeScript.\n"
    "- Provide correct, idiomatic code with proper error handling.\n"
    "- Explain your reasoning before code solutions.\n"
    "- For code review: identify bugs, security issues, and performance problems.\n"
    "- Follow PEP 8 for Python and standard conventions for JavaScript/TypeScript.\n"
    "- State assumptions about API versions when unsure."
)

LANGUAGE_INFO = {
    "python": {"extensions": [".py"], "prompt_prefix": "# Python code:\n"},
    "javascript": {"extensions": [".js", ".jsx", ".mjs"], "prompt_prefix": "// JavaScript code:\n"},
    "typescript": {"extensions": [".ts", ".tsx"], "prompt_prefix": "// TypeScript code:\n"},
}


@torch.no_grad()
def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, device: str = "cpu") -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.cfg["max_seq_len"] else x[:, -model.cfg["max_seq_len"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == tokenizer.token_to_id("<eos>"):
            break
    generated = tokenizer.decode(x[0].tolist())
    return generated[len(prompt):] if generated.startswith(prompt) else generated


def format_prompt(message: str, language: str, context: str = "") -> str:
    parts = [f"<|system|>{CODE_SYSTEM_PROMPT}\n"]
    if context:
        lang_label = language if language != "auto" else "unknown"
        parts.append(f"<|user|>\nContext ({lang_label}):\n```\n{context}\n```\n\nQuestion: {message}\n<|assistant|>\n")
    else:
        parts.append(f"<|user|>\n{message}\n<|assistant|>\n")
    return "".join(parts)


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

    with gr.Blocks(title="Quantsmind Coder", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Quantsmind Coder — AI Programming Assistant")
        gr.Markdown("Python & JavaScript/TypeScript specialist. Code generation, debugging, review, and explanation.")

        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=["auto", "python", "javascript", "typescript"],
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

    p = argparse.ArgumentParser(description="Quantsmind Coder — Code Chat UI")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model, model_cfg = load_model(args.ckpt, device)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print(f"Starting Quantsmind Coder on http://localhost:{args.port}")
    ui = create_code_chat_ui(model, tokenizer, device)
    ui.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
