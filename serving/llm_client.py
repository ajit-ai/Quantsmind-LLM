import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_base=os.environ.get("QM_API_BASE", ""),
            api_key=os.environ.get("QM_API_KEY", ""),
            model=os.environ.get("QM_MODEL", ""),
        )

    def configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)


SYSTEM_PROMPT = (
    "You are Quantsmind, a financial knowledge assistant.\n"
    "- Explain financial and market concepts clearly for retail investors.\n"
    "- Use India-first conventions (Rupees, crore/lakh, NSE/BSE, SEBI) unless asked otherwise.\n"
    "- Show formulas and worked numeric examples where useful.\n"
    "- You provide educational information only; never give personalized buy/sell advice."
)

DISCLAIMER = "\n\n---\n*Educational information only — not investment advice.*"


def append_disclaimer(text: str) -> str:
    if DISCLAIMER.strip() in text:
        return text
    return text + DISCLAIMER


def build_messages(history: list, message: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            msgs.append({"role": "user", "content": str(turn[0])})
            msgs.append({"role": "assistant", "content": str(turn[1])})
    msgs.append({"role": "user", "content": message})
    return msgs


def client_stream(api_base: str, api_key: str, model: str, messages: list):
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


DEMO_ANSWERS = [
    (
        ("pe ", "pe?", "p/e", "price-to-earnings"),
        "**P/E Ratio (Price-to-Earnings)**\n\n"
        "P/E = Share Price / Earnings Per Share (EPS)\n\n"
        "Example: If a stock trades at Rs.500 and its EPS is Rs.25, "
        "P/E = 500 / 25 = **20x**. Investors pay Rs.20 for every Re.1 of annual earnings.\n\n"
        "Rules of thumb: high P/E implies growth expectations; compare within the same sector."),
    (
        ("cagr",),
        "**CAGR (Compound Annual Growth Rate)**\n\n"
        "CAGR = (End Value / Start Value)^(1/years) - 1\n\n"
        "Example: Rs.10,000 grows to Rs.16,000 in 3 years:\n"
        "CAGR = (16000/10000)^(1/3) - 1 = **~17%** per year."),
    (
        ("sip",),
        "**SIP (Systematic Investment Plan)**\n\n"
        "A fixed amount invested at regular intervals, averaging out purchase cost "
        "(rupee-cost averaging).\n\n"
        "Example: Rs.5,000/month at 12% p.a. for 10 years grows to roughly **Rs.11.6 lakh** "
        "(invested Rs.6 lakh)."),
]


def demo_stream(message: str):
    low = " " + message.lower() + " "
    body = None
    for keys, ans in DEMO_ANSWERS:
        if any(k in low for k in keys):
            body = ans
            break
    if body is None:
        body = (
            "**Quantsmind demo mode** — no LLM backend configured.\n\n"
            "Set these environment variables to connect any OpenAI-compatible model:\n"
            "`QM_API_BASE`, `QM_API_KEY`, `QM_MODEL`\n\n"
            "Try asking about: P/E ratio, CAGR, SIP."
        )
    for i in range(0, len(body), 24):
        yield body[: i + 24]
