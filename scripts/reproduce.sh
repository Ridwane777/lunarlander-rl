#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — LunarLander-v3
# =============================================================================
# Régénère TOUS les résultats du projet en une seule commande :
#   1. Vérifie l'environnement Python
#   2. Installe les dépendances
#   3. Entraîne tous les agents (5 seeds × 5 algos)
#   4. Génère le benchmark comparatif et tous les graphes
#
# Usage :
#   bash reproduce.sh                  # entraînement complet
#   bash reproduce.sh --eval-only      # benchmark uniquement (logs déjà présents)
#   bash reproduce.sh --agent dqn_soft # un seul agent
#   bash reproduce.sh --seed 0         # une seule seed
#
# Reproductibilité garantie à ±5% avec les mêmes seeds (0..4).
# =============================================================================

set -euo pipefail

# ── couleurs terminal ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── paramètres ────────────────────────────────────────────────────────────────
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

# ── répertoire racine ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  LunarLander RL — Reproduce Script${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# ── 1. vérification Python ────────────────────────────────────────────────────
info "Vérification de Python..."

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

# ── 2. dépendances ────────────────────────────────────────────────────────────
info "Installation des dépendances..."
pip install -q -r requirements.txt
success "Dépendances installées"

# Vérifie gymnasium[box2d]
python3 -c "import gymnasium; gym_v=gymnasium.__version__; print(f'gymnasium {gym_v}')" || \
  { warn "gymnasium manquant, installation..."; pip install -q "gymnasium[box2d]"; }

python3 -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null || \
  error "torch introuvable après installation"

python3 -c "import yaml" 2>/dev/null || \
  { warn "pyyaml manquant, installation..."; pip install -q pyyaml; }

# ── 3. création des dossiers ──────────────────────────────────────────────────
mkdir -p logs/train logs/eval plots videos models configs
success "Dossiers créés"

# ── 4. entraînement ───────────────────────────────────────────────────────────
if [[ "$EVAL_ONLY" == false ]]; then
  echo ""
  echo -e "${BOLD}--- Entraînement ---${NC}"

  if [[ -n "$AGENT" ]]; then
    CONFIG="configs/${AGENT}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
      error "Config introuvable : $CONFIG"
    fi
    info "Agent : $AGENT | config : $CONFIG"
    python3 train.py --config "$CONFIG" $SEED_ARG
  else
    info "Entraînement de tous les agents (5 seeds x 5 configs)..."
    for CONFIG in configs/*.yaml; do
      ALGO=$(basename "$CONFIG" .yaml)
      info "→ $ALGO ..."
      python3 train.py --config "$CONFIG" $SEED_ARG
    done
  fi

  success "Entraînement terminé — logs dans logs/train/"
fi

# ── 5. benchmark & graphes ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}--- Benchmark & graphes ---${NC}"

if ls models/*.pth &>/dev/null; then
  python3 eval.py
  success "Benchmark terminé — graphes dans plots/benchmark/"
else
  warn "Aucun modèle .pth dans models/ — lance d'abord l'entraînement."
fi

# ── 6. résumé ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  Reproduction terminée avec succès.${NC}"
echo -e "${BOLD}  Résultats :${NC}"
echo -e "    logs/train/    — CSV d'apprentissage"
echo -e "    logs/eval/     — CSV des tests d'évaluation déterministe"
echo -e "    models/        — Poids des modèles (.pth)"
echo -e "    videos/train/  — Vidéos d'entraînement"
echo -e "    videos/evals/  — Vidéos de tests"
echo -e "    plots/         — Courbes d'entraînement et benchmark comparatif"
echo -e "${BOLD}============================================================${NC}"
echo ""
