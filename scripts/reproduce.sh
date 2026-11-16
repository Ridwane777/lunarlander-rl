#!/usr/bin/env bash
# Regenerates every result of the project from scratch:
#   1. trains the learning agents (configs/*.yaml, 5 seeds each)
#   2. evaluates them and the two baselines on the same 100 episodes
#   3. redraws the training curves and the benchmark plots
#
# Usage:
#   bash scripts/reproduce.sh                   # everything (several hours on a CPU)
#   bash scripts/reproduce.sh --eval-only       # benchmark only, from the models in models/
#   bash scripts/reproduce.sh --agent dqn_hard  # train a single configuration
#   bash scripts/reproduce.sh --seed 0          # a single seed per configuration
set -euo pipefail

EVAL_ONLY=false
AGENT=""
SEED_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --eval-only) EVAL_ONLY=true; shift ;;
    --agent)     AGENT="$2"; shift 2 ;;
    --seed)      SEED_ARGS=(--seed "$2"); shift 2 ;;
    -h|--help)   sed -n '2,11p' "$0"; exit 0 ;;
    *)           echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -c "import gymnasium, torch, yaml" 2>/dev/null || {
  echo "missing dependencies: pip install -r requirements.txt" >&2
  exit 1
}

if [[ "$EVAL_ONLY" == false ]]; then
  if [[ -n "$AGENT" ]]; then
    configs=("configs/${AGENT}.yaml")
    [[ -f "${configs[0]}" ]] || { echo "config not found: ${configs[0]}" >&2; exit 1; }
  else
    configs=(configs/*.yaml)
  fi
  for config in "${configs[@]}"; do
    echo "== training $(basename "$config" .yaml)"
    python3 train.py --config "$config" "${SEED_ARGS[@]}"
  done
fi

echo "== evaluation and benchmark"
python3 eval.py

echo "done: logs/ (CSV), models/ (weights), plots/ (curves, benchmark), videos/ (MP4)"
