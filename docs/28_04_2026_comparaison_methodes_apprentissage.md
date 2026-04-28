# Comparaison des methodes d'apprentissage

## Objectif

Comparer trois methodes de reinforcement learning sur `LunarLander-v3` pour identifier la meilleure:
- Q-Learning
- REINFORCE
- DQN (soft update)

## Donnees utilisees

Les resultats proviennent des logs CSV:
- `logs/q_learning.csv`
- `logs/reinforce.csv`
- `logs/dqn_soft.csv`

Les visualisations ont été faites dans:
- `plots/method_comparison/`

## Metriques de comparaison

Metriques principales:
- `mean_last_100` (moyenne des 100 derniers episodes) -> critere principal
- episode de resolution (seuil 200)
- taux de `landed`, `crash`, `timeout`
- reward moyen global

Resultats (extraits de `logs/summary_metrics.csv`):

| Methode | Episodes | Mean reward | Mean last 100 | Solved episode | Landed rate | Crash rate |
|---|---:|---:|---:|---|---:|---:|
| DQN (soft) | 187 | 110.71 | 202.30 | 186 | 70.6% | 13.4% |
| REINFORCE | 687 | -52.68 | 64.33 | not_solved | 17.8% | 66.2% |
| Q-Learning | 867 | -98.99 | 7.07 | not_solved | 17.5% | 82.2% |

## Figures produites

- `plots/method_comparison/01_rewards_trend.png`: rewards bruts + moyenne mobile
- `plots/method_comparison/02_mean100.png`: moyenne glissante sur 100 episodes
- `plots/method_comparison/03_rewards_boxplot.png`: comparaison simple (mean_last_100 + landed rate)
- `plots/method_comparison/04_termination_causes.png`: causes de fin d'episode

## Conclusion

La meilleure methode sur cette campagne est **DQN (soft)**:
- seule methode qui atteint le seuil de resolution (`mean_last_100 >= 200`)
- meilleur taux de succes (`landed`)
- meilleure performance finale et meilleure convergence

REINFORCE est deuxieme mais ne franchit pas le seuil de 200, et Q-Learning est le moins performant sur ces essais.
