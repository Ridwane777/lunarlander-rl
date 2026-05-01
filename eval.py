"""
eval.py — LunarLander RLLunarLander-v3
================================================
Évalue les modèles .pth entraînés dans un environnement de test
et génère un benchmark comparatif à partir de ces évaluations.

Usage:
    python eval.py                      # tous les modèles disponibles dans models/
    python eval.py --models models/dqn_soft_seed*.pth
    python eval.py --out plots/benchmark
"""

import argparse
import glob
import os
import sys
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.evals.eval_core import evaluate_model
from src.evals.eval_metrics import algo_from_path, load_all, compute_metrics
from src.evals.eval_plots import (
    plot_mean_ci,
    plot_distribution,
    plot_termination,
    plot_success_rate,
    plot_episode_length
)
from src.evals.eval_utils import save_summary_csv, print_verdict

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation des modèles .pth et benchmark."
    )
    parser.add_argument("--models", nargs="*",
        help="Chemins des .pth à évaluer. Absent = tous les models/*.pth")
    parser.add_argument("--n_episodes", type=int, default=100,
        help="Nombre d'épisodes de test par modèle.")
    parser.add_argument("--out", default="plots/benchmark",
        help="Dossier de sortie pour les graphes (défaut: plots/benchmark)")
    args = parser.parse_args()

    if args.models:
        model_paths = args.models
    else:
        model_paths = sorted(glob.glob(os.path.join(ROOT, "models", "*.pth")))

    if not model_paths:
        print("Aucun modèle (.pth) trouvé. Lancez d'abord l'entraînement (python train.py).")
        sys.exit(1)

    print(f"\nModèles à évaluer ({len(model_paths)}) :")
    for p in model_paths:
        print(f"  {p}")

    eval_logs_paths = []

    # Boucle d'évaluation
    for path in model_paths:
        algo = algo_from_path(path)
        m = re.search(r"seed(\d+)", os.path.basename(path))
        seed = int(m.group(1)) if m else 0
        
        config_path = os.path.join(ROOT, "configs", f"{algo}.yaml")
        if not os.path.exists(config_path):
            if algo == "dqn":
                config_path = os.path.join(ROOT, "configs", "dqn_hard.yaml")
            if not os.path.exists(config_path):
                print(f"[ATTENTION] Config introuvable pour {path}, passage.")
                continue
            
        log_path = evaluate_model(path, config_path, algo, seed, n_episodes=args.n_episodes)
        if log_path:
            eval_logs_paths.append(log_path)

    if not eval_logs_paths:
        print("Aucune évaluation n'a abouti.")
        sys.exit(1)

    # Analyses et graphes
    data    = load_all(eval_logs_paths)
    metrics = compute_metrics(data)
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGénération des graphes dans {out_dir}/")
    plot_mean_ci(metrics, out_dir)
    plot_distribution(data, out_dir)
    plot_termination(metrics, out_dir)
    plot_success_rate(metrics, out_dir)
    plot_episode_length(data, out_dir)
    save_summary_csv(metrics, out_dir)

    print_verdict(metrics)

if __name__ == "__main__":
    main()
