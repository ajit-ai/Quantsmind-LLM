from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_parse():
    files = sorted(CONFIG_DIR.rglob("*.yaml"))
    assert len(files) >= 5
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f.name


def test_model_config_required_keys():
    required = {"d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_dim", "vocab_size"}
    for f in (CONFIG_DIR / "model").glob("*.yaml"):
        cfg = yaml.safe_load(f.read_text(encoding="utf-8"))["model"]
        missing = required - set(cfg)
        assert not missing, f"{f.name}: {missing}"
        assert cfg["n_heads"] % cfg["n_kv_heads"] == 0


def test_promotion_yaml_rules():
    promo = Path(__file__).resolve().parents[1] / "evaluation" / "promotion.yaml"
    data = yaml.safe_load(promo.read_text(encoding="utf-8"))
    assert "rules" in data and len(data["rules"]) >= 3
