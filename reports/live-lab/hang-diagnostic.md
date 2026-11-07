# Taksh Live Lab Hang Diagnostic

**Completed:** 2026-10-21T08:54:48+05:30  
**Final disposition:** `INFERENCE_INCORRECT`  
**Transport:** PASS — HTTP 200 in 2,100 ms  
**Answer correctness:** FAIL — expected `4`, received unrelated code-like text

No browser UI was implemented, no 40-prompt suite was run, and nothing was
committed, tagged, pushed, merged, or published. No checkpoint or weight changed.

## 1. Repository and environment

- Repository: `F:\Codes\Git\Quantsmind-LLM`
- Branch: `develop`; HEAD `cb9b81e` (`Merge develop: synchronize v1.0.0 source release`)
- Git status at start: clean. Tag `v1.0.0` remains at `a156892`.
- No `serving/live_lab.py`, `tests/test_live_lab.py`, or `docs/LIVE_LAB.md` found.
- No previous Live Lab source, traceback, or report was found in the repository.
- `reports/live-lab/` did not exist before this diagnostic.
- OS: Windows-11-10.0.26300-SP0; Python 3.14.7; PyTorch 2.13.0+cpu.
- CUDA unavailable; torch threads 4; total RAM 4,005 MB.
- Free RAM: 341 MB initially, 310 MB before inference, 283 MB afterward.
- No dependency was installed.

## 2. Commands executed

```powershell
git status --short
git branch --show-current
git log --oneline -3
Get-CimInstance Win32_Process -Filter "Name='python.exe'"
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 7860,7861,8002,8012 }
Invoke-RestMethod -Uri 'http://127.0.0.1:8012/health' -TimeoutSec 10
Get-Process -Id 12688
Get-CimInstance Win32_OperatingSystem
Get-Item runs\m7_sft\checkpoint_best.pt
Get-FileHash runs\m7_sft\checkpoint_best.pt -Algorithm SHA256
Get-Content C:\Users\Lenovo\AppData\Local\Temp\opencode\taksh-live-lab\server.out.log -Tail 40
Get-Content C:\Users\Lenovo\AppData\Local\Temp\opencode\taksh-live-lab\server.err.log -Tail 40
Invoke-WebRequest -Uri 'http://127.0.0.1:8012/v1/chat/completions' `
  -Method Post -ContentType 'application/json' -Body $body `
  -TimeoutSec 60 -UseBasicParsing
```

Inference request: prompt `What is 2 + 2? Reply with only the number.`,
`max_tokens=16`, `temperature=0.0`, `top_p=1.0`, `seed=7`,
`stream=false`, model `qm-code-m3-9m`.

## 3. Startup status and duration

An M6/M7 server was already running before this diagnostic:

- PID `12688`, started `2026-10-21 08:42:27`
- `python -m serving.openai_server --ckpt runs/m7_sft/checkpoint_best.pt --tokenizer tokenizer/qm_code_bpe_v0.json --host 127.0.0.1 --port 8012`
- Listener `127.0.0.1:8012` only; health PASS within the 10-second deadline
- Health: `{"status":"ok","version":"1.0.0","model_loaded":true,"model_id":"qm-code-m3-9m","context_window":512}`

The server was not restarted. Process creation and the load log are approximately
21 seconds apart (`08:42:27` to `08:42:48`), within the 90-second startup limit.
No new process was started, avoiding a duplicate checkpoint load under low memory.
Exit code is not applicable because the process is still running.

## 4. Checkpoint load observations

- Checkpoint: `runs/m7_sft/checkpoint_best.pt`, 112,414,501 bytes
- SHA-256: `421C218D626524FFA8266A5F5C5CCB06FA743D3CCF8E75CC036B1047478B4DD9`
- Tokenizer: `tokenizer/qm_code_bpe_v0.json`, 410,135 bytes
- Step 120; 9,359,104 parameters; model `qm-code-m3-9m`
- Vocab: model 5,822 / tokenizer 5,822; context 512; device CPU
- Loader attempted `weights_only=True`, fell back to `weights_only=False` after an
  `UnpicklingError`, and logged the fallback. The path was operator-supplied.
- After inference: private bytes ~797 MB, working set ~78 MB.

## 5. Inference status and duration

- `POST /v1/chat/completions`; HTTP 200; latency 2,100 ms
- Prompt tokens 103; completion tokens 16; total 119; `finish_reason=length`
- Transport success: yes. Answer correct: no. Raw output was not rewritten.
- Exact content: `"    :param _pytest.abc.\n    \"\"\"\n    :"`
- Expected `4`; output does not match.
- Verbatim response: `reports/live-lab/inference-response-raw.json`.
- This preserves the published M7 finding: HTTP success does not imply usefulness.

## 6. Errors and timeouts

None during current startup, health, or inference. No current timeout occurred.
No previous-attempt logs, tracebacks, or source were found, so those three hangs
could not be reconstructed. Health was bounded at 10 s and inference at 60 s;
no retry was performed.

## 7. Memory observations

| Point | Free RAM | Server private bytes | Server working set |
|---|---:|---:|---:|
| Before inference | 310 MB | ~802 MB | ~30 MB |
| After inference | 283 MB | ~797 MB | ~78 MB |

Memory pressure is real, but the 16-token request completed in 2.1 s. There is no
evidence memory alone caused the earlier hangs, though a browser UI plus another
Python process could add pressure.

## 8. Root cause

**Classification: `UNKNOWN`** (layer evidence points away from M6/M7).

Supporting evidence:

- The existing server loaded the checkpoint and answered health within limits.
- One real 16-token inference returned HTTP 200 in 2.1 s.
- The server logs contain no crash, deadlock, or timeout for this path.
- Only one Python process and one relevant listener exist; no competing UI process.
- No artifacts from the three failed attempts survive, so the exact blocking call
  cannot be identified.

Missing evidence: previous attempt logs, stack traces, process IDs, terminal output,
and the prior UI implementation itself. Do not attribute the hangs to checkpoint
loading or inference; both were exercised successfully here. The most likely layer
is the discarded UI/tool workflow, but that is a hypothesis, not a proven cause.

## 9. Recommended smallest corrective action

Do not reuse the discarded UI/tool workflow. Continue only with the already-verified
M6 server plus a thin client that enforces independent per-request deadlines and
writes the report incrementally. Before any future UI attempt, free memory or stop
unused processes. This diagnostic stops here and performs no UI work.

## 10. Files changed

- Created `reports/live-lab/hang-diagnostic.md`
- Created `reports/live-lab/hang-diagnostic.json`
- Created `reports/live-lab/inference-response-raw.json`

No tracked file was modified. No Git operation was performed.

## 11. Cleanup status

No process was started by this diagnostic. The pre-existing Taksh server PID
`12688` (the process verified above) was stopped after evidence capture; its
listener on `127.0.0.1:8012` is no longer present. No unrelated process was
terminated, and no background job remains.
