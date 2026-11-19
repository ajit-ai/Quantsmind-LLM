import random

TEMPLATES = [
    "{name} ({ticker}) closed at {px:.2f} ({chg}{pct:.1f}%) on {date}.",
    "The {index} index moved {chg_words} by {pts} points amid {topic} concerns.",
    "Quarterly revenue was Rs.{rev} crore versus Rs.{prev} crore in the prior period.",
    "Analysts set a target price of Rs.{tp:.2f} with a {rating} rating.",
    "Trading volume reached {vol:,} shares, up {pctv:.1f}% from the previous session.",
    "The company reported EPS of {eps:.2f}, beating consensus estimates of {est:.2f}.",
    "Yields on the 10-year benchmark note rose to {yld:.2f} percent.",
    "Investors rotated into {sector} stocks as inflation cooled to {infl:.1f}%.",
]

NAMES = ["Infotech Ltd", "Bharat Motors", "Zen Pharma", "Delta Energy", "Apex Bank", "Orion Metals"]
TICKERS = ["AAPL", "MSFT", "RELIANCE", "TCS", "HDFCBANK", "INFY", "NIFTY50", "SENSEX"]
INDICES = ["NIFTY50", "SENSEX", "NASDAQ", "DOW"]
TOPICS = ["rate", "inflation", "earnings", "liquidity", "currency"]
SECTORS = ["IT", "banking", "pharma", "energy", "FMCG"]
CHG_WORDS = {"+": "gained", "-": "fell"}
RATINGS = ["buy", "hold", "accumulate", "sell"]


def _line(rng: random.Random) -> str:
    t = rng.choice(TEMPLATES)
    chg = rng.choice("+-")
    return t.format(
        name=rng.choice(NAMES),
        ticker=rng.choice(TICKERS),
        px=rng.uniform(50, 4000),
        chg=chg,
        pct=rng.uniform(0.05, 9.9),
        date=f"20{rng.randint(24, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        index=rng.choice(INDICES),
        pts=rng.randint(10, 800),
        rev=rng.randint(500, 90000),
        prev=rng.randint(500, 90000),
        tp=rng.uniform(100, 5000),
        rating=rng.choice(RATINGS),
        vol=rng.randint(10_000, 5_000_000),
        pctv=rng.uniform(0.1, 40),
        eps=rng.uniform(0.5, 120),
        est=rng.uniform(0.5, 120),
        yld=rng.uniform(6.0, 8.0),
        topic=rng.choice(TOPICS),
        sector=rng.choice(SECTORS),
        infl=rng.uniform(3.0, 7.5),
        chg_words=CHG_WORDS[chg],
    )


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Generate synthetic sample corpus for smoke tests")
    p.add_argument("--out", default="data/sample_corpus.txt")
    p.add_argument("--lines", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.lines):
            f.write(_line(rng) + "\n")
            if rng.random() < 0.15:
                f.write(
                    "def metric(xs):\n"
                    "    mu = sum(xs) / len(xs)\n"
                    f"    return mu / (sum(abs(x - mu) for x in xs) / len(xs)) + {rng.uniform(-1, 1):.4f}\n\n"
                )
    print(f"wrote {args.lines} lines to {out}")


if __name__ == "__main__":
    main()
