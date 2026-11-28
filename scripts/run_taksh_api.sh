#!/usr/bin/env bash
# TAKSH Local API — Linux/macOS launcher.
#
# Starts the OpenAI-compatible local coding backend on 127.0.0.1. No external
# service is contacted; weights are loaded once from the local checkpoint.
#
# Usage:
#   ./scripts/run_taksh_api.sh
#   ./scripts/run_taksh_api.sh --port 8010 --strict-history
#   QM_API_KEY=secret ./scripts/run_taksh_api.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${TAKSH_CHECKPOINT:-runs/i4_aligned_sft_long/checkpoint_last.pt}"
TOKENIZER="${TAKSH_TOKENIZER:-tokenizer/taksh_code_bpe_v1.json}"
HOST="127.0.0.1"
PORT="8002"
STRICT_HISTORY=0
ALLOW_REMOTE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --strict-history) STRICT_HISTORY=1; shift ;;
    --allow-remote) ALLOW_REMOTE=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for path in "$CHECKPOINT" "$TOKENIZER"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path (relative to $REPO_ROOT)." >&2
    exit 1
  fi
done

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

ARGS=(-m serving.openai_server --ckpt "$CHECKPOINT" --tokenizer "$TOKENIZER"
      --host "$HOST" --port "$PORT")
if [[ "$STRICT_HISTORY" -eq 0 ]]; then
  ARGS+=(--flatten-history)
fi
if [[ "$ALLOW_REMOTE" -eq 1 ]]; then
  ARGS+=(--allow-remote)
fi

echo
echo "TAKSH Local API"
echo "  base URL   : http://${HOST}:${PORT}/v1"
echo "  checkpoint : $CHECKPOINT"
echo "  tokenizer  : $TOKENIZER"
if [[ -n "${QM_API_KEY:-}" ]]; then
  echo "  auth       : bearer key set"
else
  echo "  auth       : off (loopback only)"
fi
if [[ "$STRICT_HISTORY" -eq 1 ]]; then
  echo "  history    : strict contract (multi-turn refused)"
else
  echo "  history    : flattened compatibility mode"
fi
echo
echo "Capability is not claimed: the frozen Native v0.1 baseline is 0/20."
echo "Press Ctrl+C to stop."
echo

exec "$PYTHON" "${ARGS[@]}"
