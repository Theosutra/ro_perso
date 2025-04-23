"""
benchmark_runner.py

Script pour lancer une série d'expérimentations automatisées sur des graphes
de différentes tailles et configurations, et collecter les résultats dans un CSV.
"""

import os
import time
import itertools
from datetime import datetime
import pandas as pd
import argparse
import random

# Importer nos modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_generator import TSPGraphGenerator
from genetic_solver import GeneticTSPSolver
from results_logger import ResultsLogger
from runner import run_single_experiment, parse_arguments


def run_benchmark(output_dir='benchmark_results',
                  city_sizes=[10, 15, 20, 30, 50],
                  dependency_ratios=[0.1, 0.2, 0.3],
                  blocking_ratios=[0.05, 0.1, 0.15],
                  vehicle_capacities=[500, 1000, 1500],
                  population_sizes=[100, 200, 300],
                  mutation_rates=[0.05, 0.1, 0.2],
                  runs_per_config=3,
                  max_configs=None,
                  seed=None):
    """
    Exécute une série de benchmarks en variant les paramètres et collecte les résultats.
    
    Args:
        output_dir: Répertoire pour les résultats
        city_sizes: Liste des tailles de villes à tester
        dependency_ratios: Liste des ratios de dépendances à tester
        blocking_ratios: Liste des ratios de blocage à tester
        vehicle_capacities: Liste des capacités de véhicule à tester
        population_sizes: Liste des tailles de population à tester
        mutation_rates: Liste des taux de mutation à tester
        runs_per_config: Nombre d'exécutions par configuration
        max_configs: Nombre maximal de configurations à tester (None = toutes)
        seed: Graine aléatoire pour la reproductibilité
    """
    # Créer le répertoire de sortie
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialiser le logger pour collecter les résultats
    logger = ResultsLogger(output_dir=output_dir)
    
    # Horodatage pour le nom de fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Générer toutes les combinaisons de paramètres
    all_configs = list(itertools.product(
        city_sizes,
        dependency_ratios,
        blocking_ratios,
        vehicle_capacities,
        population_sizes,
        mutation_rates
    ))
    
    # Limiter le nombre de configurations si spécifié
    if max_configs and max_configs < len(all_configs):
        random.shuffle(all_configs)
        all_configs = all_configs[:max_configs]
    
    print(f"Lancement du benchmark avec {len(all_configs)} configurations différentes")
    print(f"Chaque configuration sera exécutée {runs_per_config} fois")
    print(f"Total de {len(all_configs) * runs_per_config} exécutions prévues")
    
    # Initialiser un dataframe pour collecter les résultats
    results_df = pd.DataFrame()
    
    # Initialiser les arguments par défaut
    args = parse_arguments()
    
    # Compteur pour suivre l'avancement
    total_runs = len(all_configs) * runs_per_config
    completed_runs = 0
    
    # Heure de début pour estimer le temps restant
    start_time = time.time()
    
    for config_idx, config in enumerate(all_configs):
        num_cities, dep_ratio, block_ratio, vehicle_cap, pop_size, mut_rate = config
        
        print(f"\n--- Configuration {config_idx+1}/{len(all_configs)} ---")
        print(f"Villes: {num_cities}, Dépendances: {dep_ratio}, Blocages: {block_ratio}")
        print(f"Capacité: {vehicle_cap}, Population: {pop_size}, Mutation: {mut_rate}")
        
        # Mettre à jour les arguments avec cette configuration
        args.num_cities = num_cities
        args.dependency_ratio = dep_ratio
        args.blocking_ratio = block_ratio
        args.vehicle_capacity = vehicle_cap
        args.population_size = pop_size
        args.mutation_rate = mut_rate
        args.max_iterations = 500  # Valeur fixe pour toutes les exécutions
        args.early_stopping = 50   # Valeur fixe pour toutes les exécutions
        args.geographical = True   # Utiliser des coordonnées géographiques
        args.visualize = False     # Désactiver la visualisation pour le benchmark
        args.save_graph = False    # Ne pas sauvegarder les graphes individuels
        
        # Exécuter plusieurs fois avec cette configuration
        for run in range(runs_per_config):
            # Générer une graine unique pour cette exécution
            if seed is not None:
                run_seed = seed + config_idx * 1000 + run
            else:
                run_seed = random.randint(1, 100000)
            
            run_id = f"c{config_idx+1}_r{run+1}"
            
            # Exécuter l'expérimentation
            try:
                result = run_single_experiment(args, seed=run_seed, run_id=run_id)
                logger.add_result(result)
                
                # Extraire les métriques clés pour le CSV
                row = {
                    'run_id': run_id,
                    'num_cities': num_cities,
                    'dependency_ratio': dep_ratio,
                    'blocking_ratio': block_ratio,
                    'vehicle_capacity': vehicle_cap,
                    'population_size': pop_size,
                    'mutation_rate': mut_rate,
                    'total_cost': result['total_cost'],
                    'iterations': result['performance_metrics']['iterations'],
                    'execution_time': result['performance_metrics']['execution_time_sec'],
                    'dependencies_respected': result['constraints']['dependencies_respected'],
                    'capacity_respected': result['constraints']['capacity_respected'],
                    'total_load': result['vehicle_load_stats']['total_volume_delivered'],
                    'max_load': result['vehicle_load_stats'].get('maximum_load', 0),
                    'seed': run_seed
                }
                
                # Ajouter à notre dataframe
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                
                # Sauvegarder les résultats partiels après chaque exécution
                csv_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
                results_df.to_csv(csv_path, index=False)
                
                # Mettre à jour le compteur et afficher la progression
                completed_runs += 1
                elapsed_time = time.time() - start_time
                avg_time_per_run = elapsed_time / completed_runs
                estimated_total_time = avg_time_per_run * total_runs
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"  Exécution {run+1}/{runs_per_config} terminée. "
                      f"Progression: {completed_runs}/{total_runs} "
                      f"({completed_runs/total_runs*100:.1f}%)")
                print(f"  Temps estimé restant: {remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"  ERREUR dans l'exécution {run_id}: {str(e)}")
                # Continuer avec la prochaine exécution
    
    # Sauvegarder tous les résultats dans le CSV final
    final_csv_path = os.path.join(output_dir, f"benchmark_final_{timestamp}.csv")
    results_df.to_csv(final_csv_path, index=False)
    
    # Sauvegarder également les résultats complets avec le logger
    logger.save_to_json(filename=f"benchmark_full_{timestamp}.json")
    
    print(f"\nBenchmark terminé! Résultats sauvegardés dans {final_csv_path}")
    
    # Afficher un résumé des résultats
    print("\n=== Résumé des résultats ===")
    print(f"Nombre total d'exécutions réussies: {len(results_df)}")
    print(f"Coût moyen: {results_df['total_cost'].mean():.2f}")
    print(f"Temps d'exécution moyen: {results_df['execution_time'].mean():.2f} secondes")
    print(f"Taux de respect des dépendances: {results_df['dependencies_respected'].mean()*100:.1f}%")
    print(f"Taux de respect de la capacité: {results_df['capacity_respected'].mean()*100:.1f}%")
    
    # Retourner le dataframe pour analyse ultérieure
    return results_df


def parse_benchmark_args():
    """Parse les arguments spécifiques au benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark pour TSP avec contraintes hybrides')
    
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Répertoire de sortie pour les résultats')
    
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Nombre maximal de configurations à tester (None = toutes)')
    
    parser.add_argument('--runs_per_config', type=int, default=3,
                        help='Nombre d\'exécutions par configuration')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Graine aléatoire pour la reproductibilité')
    
    parser.add_argument('--small', action='store_true',
                        help='Exécuter un benchmark plus petit (moins de configurations)')
    
    parser.add_argument('--medium', action='store_true',
                        help='Exécuter un benchmark moyen')
    
    parser.add_argument('--large', action='store_true',
                        help='Exécuter un benchmark complet (beaucoup de configurations)')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_benchmark_args()
    
    # Définir les paramètres de benchmark selon la taille
    if args.small:
        # Petit benchmark: ~18 configurations (54 exécutions)
        run_benchmark(
            output_dir=args.output_dir,
            city_sizes=[10, 20],
            dependency_ratios=[0.1, 0.2],
            blocking_ratios=[0.1],
            vehicle_capacities=[1000],
            population_sizes=[100, 200],
            mutation_rates=[0.1],
            runs_per_config=args.runs_per_config,
            max_configs=args.max_configs,
            seed=args.seed
        )
    elif args.medium:
        # Benchmark moyen: ~108 configurations (324 exécutions)
        run_benchmark(
            output_dir=args.output_dir,
            city_sizes=[10, 20, 30],
            dependency_ratios=[0.1, 0.2, 0.3],
            blocking_ratios=[0.05, 0.1],
            vehicle_capacities=[800, 1200],
            population_sizes=[100, 200],
            mutation_rates=[0.05, 0.1],
            runs_per_config=args.runs_per_config,
            max_configs=args.max_configs,
            seed=args.seed
        )
    elif args.large:
        # Grand benchmark: complet avec toutes les combinaisons
        run_benchmark(
            output_dir=args.output_dir,
            city_sizes=[10, 15, 20, 30, 50],
            dependency_ratios=[0.1, 0.2, 0.3],
            blocking_ratios=[0.05, 0.1, 0.15],
            vehicle_capacities=[500, 1000, 1500],
            population_sizes=[100, 200, 300],
            mutation_rates=[0.05, 0.1, 0.2],
            runs_per_config=args.runs_per_config,
            max_configs=args.max_configs,
            seed=args.seed
        )
    else:
        # Configuration par défaut: benchmark personnalisé
        run_benchmark(
            output_dir=args.output_dir,
            city_sizes=[10, 20, 30],  # Petite, moyenne et grande taille
            dependency_ratios=[0.1, 0.2],  # Peu et beaucoup de dépendances
            blocking_ratios=[0.05, 0.1],  # Peu et beaucoup de routes bloquées
            vehicle_capacities=[800, 1200],  # Petite et grande capacité
            population_sizes=[100, 200],  # Petite et grande population
            mutation_rates=[0.05, 0.1],  # Faible et forte mutation
            runs_per_config=args.runs_per_config,
            max_configs=args.max_configs,
            seed=args.seed
        )