"""Resume support for training/train_mm.py (mid-run checkpoints).

Fast, hermetic: exercises save/load round-trip on a tiny model (not the full
9.5M QMMultimodal stack) plus CLI defaults. The long CPU training run itself
is the end-to-end proof.
"""
import torch

from training.train_mm import build_arg_parser, load_mm_resume, save_mm_checkpoint


def _tiny():
    model = torch.nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optim


def test_save_load_resume_roundtrip(tmp_path):
    model, optim = _tiny()
    x = torch.randn(2, 4)
    for _ in range(2):
        optim.zero_grad()
        model(x).sum().backward()
        optim.step()
    before = [p.detach().clone() for p in model.parameters()]

    path = tmp_path / "checkpoint_2.pt"
    save_mm_checkpoint(path, model, optim, 2, {"run": {}}, {"d_model": 4}, {"kind": "test"})
    assert path.is_file()

    model2, optim2 = _tiny()
    ckpt, step = load_mm_resume(path, model2, optim2)
    assert step == 2
    for a, b in zip(model2.parameters(), before):
        assert torch.equal(a.detach(), b)
    assert set(ckpt.keys()) >= {"model", "optimizer", "step", "config", "cfg", "metadata"}
    assert len(optim2.state) > 0  # optimizer state (momentum) restored


def test_parser_resume_defaults():
    args = build_arg_parser().parse_args(["--config", "configs/train/mm_m1_local.yaml"])
    assert args.resume_from is None
    assert args.ckpt_every is None
    assert args.total_steps is None
    assert args.out_dir == "runs/mm_fit_check"
