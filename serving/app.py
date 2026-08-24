import os

import gradio as gr

from serving.llm_client import (
    LLMConfig,
    append_disclaimer,
    build_messages,
    client_stream,
    demo_stream,
)

TITLE = "Quantsmind — Financial Knowledge Assistant"
DESCRIPTION = (
    "Ask anything about markets, investing, and financial concepts. "
    "India-first answers. Educational only — not investment advice."
)


def respond(message: str, history):
    if not message.strip():
        yield ""
        return
    cfg = LLMConfig.from_env()
    if not cfg.configured():
        for chunk in demo_stream(message):
            yield chunk
        return
    messages = build_messages(history, message)
    full = []
    try:
        for delta in client_stream(cfg.api_base, cfg.api_key, cfg.model, messages):
            full.append(delta)
            yield "".join(full)
    except Exception as e:
        yield f"Backend error: {e}\n\nCheck `QM_API_BASE`, `QM_API_KEY`, `QM_MODEL`."
        return
    yield append_disclaimer("".join(full))


def build_ui():
    with gr.Blocks() as ui:
        gr.ChatInterface(
            fn=respond,
            title=TITLE,
            description=DESCRIPTION,
            examples=[
                "What is a P/E ratio and how do I interpret it?",
                "Explain CAGR with an example.",
                "What happens to my SIP if markets crash?",
            ],
        )
    return ui


def main():
    port = int(os.environ.get("QM_UI_PORT", "7860"))
    cfg = LLMConfig.from_env()
    mode = f"backend={cfg.model}" if cfg.configured() else "DEMO MODE (no backend configured)"
    print(f"Starting Quantsmind UI on http://localhost:{port} [{mode}]")
    build_ui().launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
