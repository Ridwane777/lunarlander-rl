#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — LunarLander-v3
# =============================================================================
# Regenerates ALL project results with a single command:
#   1. Checks the Python environment
#   2. Installs dependencies
#   3. Trains every agent (5 seeds × 5 algos)
#   4. Builds the comparative benchmark and all plots
#
# Usage :
#   bash reproduce.sh                  # full training
#   bash reproduce.sh --eval-only      # benchmark only (logs already present)
#   bash reproduce.sh --agent dqn_soft # a single agent
#   bash reproduce.sh --seed 0         # a single seed
#
# Results reproduce within ±5% with the same seeds (0..4).
# =============================================================================

set -euo pipefail

# ── couleurs terminal ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── parameters ────────────────────────────────────────────────────────────────
EVAL_ONLY=false
AGENT=""
SEED_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --eval-only)   EVAL_ONLY=true; shift ;;
    --agent)       AGENT="$2"; shift 2 ;;
    --seed)        SEED_ARG="--seed $2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash reproduce.sh [--eval-only] [--agent dqn_soft] [--seed N]"
      exit 0 ;;
    *) error "Argument inconnu : $1" ;;
  esac
done

# ── root directory ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  LunarLander RL — Reproduce Script${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# ── 1. check Python ────────────────────────────────────────────────────
info "Checking Python..."

if ! command -v python3 &>/dev/null; then
  error "python3 introuvable. Installe Python 3.12+."
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
  error "Python >= 3.10 requis (actuel : $PY_VERSION)"
fi
success "Python $PY_VERSION"

# ── 2. dependencies ────────────────────────────────────────────────────────────
info "Installing dependencies..."
pip install -q -r requirements.txt
success "Dependencies installed"

# Check gymnasium[box2d]
python3 -c "import gymnasium; gym_v=gymnasium.__version__; print(f'gymnasium {gym_v}')" || \
  { warn "gymnasium manquant, installation..."; pip install -q "gymnasium[box2d]"; }

python3 -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null || \
  error "torch not found after installation"

python3 -c "import yaml" 2>/dev/null || \
  { warn "pyyaml manquant, installation..."; pip install -q pyyaml; }

# ── 3. create directories ──────────────────────────────────────────────────
mkdir -p logs/train logs/eval plots videos models configs
success "Directories created"

# ── 4. training ───────────────────────────────────────────────────────────
if [[ "$EVAL_ONLY" == false ]]; then
  echo ""
  echo -e "${BOLD}--- Training ---${NC}"

  if [[ -n "$AGENT" ]]; then
    CONFIG="configs/${AGENT}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
      error "Config introuvable : $CONFIG"
    fi
    info "Agent : $AGENT | config : $CONFIG"
    python3 train.py --config "$CONFIG" $SEED_ARG
  else
    info "Training all agents (5 seeds x 5 configs)..."
    for CONFIG in configs/*.yaml; do
      ALGO=$(basename "$CONFIG" .yaml)
      info "→ $ALGO ..."
      python3 train.py --config "$CONFIG" $SEED_ARG
    done
  fi

  success "Training finished — logs in logs/train/"
fi

# ── 5. benchmark & graphes ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}--- Benchmark & graphes ---${NC}"

if ls models/*.pth &>/dev/null; then
  python3 eval.py
  success "Benchmark finished — plots in plots/benchmark/"
else
  warn "No .pth model in models/ — run training first."
fi

# ── 6. summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  Reproduction completed successfully.${NC}"
echo -e "${BOLD}  Results:${NC}"
echo -e "    logs/train/    — CSV d'apprentissage"
echo -e "    logs/eval/     — deterministic evaluation CSVs"
echo -e "    models/        — model weights (.pth)"
echo -e "    videos/train/  — training videos"
echo -e "    videos/evals/  — test videos"
echo -e "    plots/         — training curves and comparative benchmark"
echo -e "${BOLD}============================================================${NC}"
echo ""
