"""Regression test for the DLCAE runtime syntax defect."""

import importlib


def test_dlcae_runtime_module_imports():
    module = importlib.import_module("taksh.dlcae.runtime")
    assert module.DLCAERuntime
