# Taksh Release Evidence

- Generated: 2026-10-21T02:35:52+00:00
- Version: `1.0.0` (final)
- Branch/commit: `develop` / `4a5a144`
- Verdict: **RELEASE GATES PASSED**
- Publication: **NOT PERFORMED BY THIS SCRIPT**

## Checks

| Step | Result | Command |
|---|---|---|
| version-consistency | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe F:\Codes\Git\Quantsmind-LLM\scripts\check_version_consistency.py` |
| build-sdist-wheel | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m build --outdir C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\build F:\Codes\Git\Quantsmind-LLM` |
| archive-metadata | PASS | `` |
| twine-check | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m twine check C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\build\taksh-1.0.0-py3-none-any.whl C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\build\taksh-1.0.0.tar.gz` |
| create-clean-venv | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m venv C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-clean` |
| install-clean-venv | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-clean\Scripts\python.exe -m pip install --no-index --no-deps C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\build\taksh-1.0.0-py3-none-any.whl` |
| installed-version-clean-venv | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-clean\Scripts\python.exe -c import importlib.metadata as m, taksh; assert m.version('taksh') == taksh.__version__ == '1.0.0'; assert 'site-packages' in taksh.__file__, taksh.__file__; print(taksh.__version__, taksh.__file__)` |
| installed-cli-clean-venv | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-clean\Scripts\taksh.exe --version` |
| create-system-venv | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m venv --system-site-packages C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-system` |
| install-system-venv | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-system\Scripts\python.exe -m pip install --no-index --no-deps C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\build\taksh-1.0.0-py3-none-any.whl` |
| installed-openai-cli-help | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-system\Scripts\taksh-openai.exe --help` |
| installed-serving | PASS | `C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\venv-system\Scripts\python.exe F:\Codes\Git\Quantsmind-LLM\scripts\verify_installed_release.py --repo F:\Codes\Git\Quantsmind-LLM --expect-version 1.0.0 --json-out C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\installed-serving.json` |
| pytest | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests -q --basetemp C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\pytest` |
| sphinx | PASS | `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe -m sphinx -W -b html F:\Codes\Git\Quantsmind-LLM\docs\source C:\Users\Lenovo\AppData\Local\Temp\taksh-release-k6hga43k\docs-html` |

## Artifacts

| File | Bytes | SHA-256 | Location |
|---|---:|---|---|
| `taksh-1.0.0-py3-none-any.whl` | 336,332 | `047ad9278dfc95cb8b6eff9b21299dd7cd7ecfe83ea117dc26e237617955d8c9` | `F:\Codes\Git\Quantsmind-LLM\dist\taksh-1.0.0-py3-none-any.whl` |
| `taksh-1.0.0.tar.gz` | 418,840 | `341225b9cf0f08ce5accef5f843e24009bc12d54395ce383643294c2f1d15a28` | `F:\Codes\Git\Quantsmind-LLM\dist\taksh-1.0.0.tar.gz` |

## Archive inspection

- Wheel: 105 members; forbidden=[]
- Source distribution: 223 members; forbidden=[]
- LICENCE and NOTICE are present in both archives and in package metadata.
- Secret scan allowlist: `tests/test_internal_ingest.py` (synthetic key-format fixtures only).
- Weights, checkpoints, tokenizer assets, private datasets and caches are excluded.

## Capability disclosure

- M7 result: 0/40 locked syntax-valid generations, 0/8 functional tasks, evidence level 0.
- Serving HTTP success is not evidence that generated output is correct.
- Package readiness and model capability have different statuses.
- Weights and training data are not part of the package.

## Publication

This evidence run does not commit, tag, push, merge, create a GitHub
release or upload to a package index. Publication is a separate
owner-authorized operation recorded in the release-readiness report.
