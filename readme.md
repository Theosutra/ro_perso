# Algorithme Génétique pour TSP avec Contraintes Hybrides

Ce projet implémente un algorithme génétique pour résoudre une version avancée du problème du voyageur de commerce (TSP) intégrant des contraintes complexes, dans un objectif de logistique optimisée à la fois économique et écologique.

## Fonctionnalités principales

- Résolution du TSP avec des contraintes hybrides:
  - **Contraintes de dépendance entre villes** - certaines villes doivent être visitées avant d'autres
  - **Routes bloquées** - certains trajets directs sont interdits
  - **Contrainte de "backpacking"** - gestion de capacité de chargement des véhicules

- Générateur de graphes aléatoires personnalisables
- Analyse de performance avec metrics détaillées
- Benchmarking pour comparaison de différentes configurations
- Visualisation des graphes et des solutions

## Structure du projet

- `graph_generator.py` - Génération de graphes aléatoires avec contraintes
- `genetic_solver.py` - Implémentation de l'algorithme génétique
- `results_logger.py` - Enregistrement et export des résultats
- `runner.py` - Point d'entrée pour les exécutions simples
- `benchmark_runner.py` - Automatisation des tests sur différentes configurations
- `analyze_benchmark.py` - Analyse statistique et visualisation des résultats

## Installation

Assurez-vous d'avoir Python 3.8+ installé, puis installez les dépendances:

```bash
pip install networkx numpy matplotlib pandas seaborn
```

## Utilisation

### Exécution simple

```bash
python runner.py --num_cities 20 --geographical --visualize
```

### Exécution avec paramètres personnalisés

```bash
python runner.py --num_cities 30 \
                --dependency_ratio 0.2 \
                --blocking_ratio 0.1 \
                --vehicle_capacity 1500 \
                --population_size 200 \
                --mutation_rate 0.1 \
                --max_iterations 1000 \
                --visualize
```

### Lancer un benchmark

```bash
# Petit benchmark (~18 configurations)
python benchmark_runner.py --small --runs_per_config 3

# Benchmark moyen (~108 configurations)
python benchmark_runner.py --medium --runs_per_config 3

# Benchmark personnalisé
python benchmark_runner.py --output_dir custom_results --runs_per_config 5 --seed 42
```

### Analyser les résultats

```bash
python analyze_benchmark.py benchmark_results/benchmark_results_20250423_120000.csv
```

## Paramètres configurables

### Paramètres du graphe
- `--num_cities` - Nombre de villes dans le graphe (défaut: 20)
- `--geographical` - Utiliser des distances géographiques au lieu d'aléatoires
- `--min_load` - Charge minimale pour une ville (défaut: 10)
- `--max_load` - Charge maximale pour une ville (défaut: 100)
- `--dependency_ratio` - Ratio de dépendances entre villes (défaut: 0.2)
- `--blocking_ratio` - Ratio de routes bloquées (défaut: 0.1)

### Paramètres de l'algorithme génétique
- `--population_size` - Taille de la population (défaut: 100)
- `--mutation_rate` - Taux de mutation (défaut: 0.1)
- `--max_iterations` - Nombre maximum d'itérations (défaut: 500)
- `--early_stopping` - Arrêt si pas d'amélioration pendant N itérations (défaut: 50)
- `--vehicle_capacity` - Capacité maximale du véhicule (défaut: 1000)

### Paramètres d'exécution
- `--num_runs` - Nombre d'exécutions avec des graines différentes (défaut: 1)
- `--output_dir` - Répertoire de sortie pour les résultats (défaut: 'results')
- `--seed` - Graine aléatoire pour reproductibilité
- `--save_graph` - Sauvegarder le graphe généré en JSON
- `--visualize` - Générer une visualisation du graphe et de la solution

## Exemples de résultats

L'algorithme génère un rapport détaillé pour chaque exécution:

```
Chemin solution: E → J → N → L → H → R → I → G → P → S → K → A → F → D → C → B → Q → M → O → T → E
Coût total: 312.12
Temps d'exécution: 0.42 secondes
Nombre d'itérations: 172
```

Les statistiques incluent:
- Respect des contraintes (dépendances et capacité)
- Charge totale et maximale du véhicule
- Capacité restante
- Performance de l'algorithme

## Suggestions d'amélioration

Pour des solutions respectant mieux les contraintes:
- Augmenter la capacité du véhicule (`--vehicle_capacity`)
- Augmenter la taille de population (`--population_size`)
- Augmenter le nombre d'itérations (`--max_iterations`)
- Ajuster le taux de mutation (`--mutation_rate`)

## Contributeurs

Ce projet a été réalisé dans le cadre d'un cours de recherche opérationnelle.

## Licence

MIT