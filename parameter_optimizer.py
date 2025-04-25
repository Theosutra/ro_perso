"""
parameter_optimizer.py

Script pour optimiser automatiquement les paramètres de l'algorithme génétique VRP
en effectuant une recherche sur différents taux de mutation.
"""

import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json
import random
from typing import Dict, List, Tuple, Any

# Importer les modules VRP
from vrp_graph_generator import VRPGraphGenerator
from vrp_genetic_solver import GeneticVRPSolver
from results_logger import ResultsLogger


def parse_optimizer_args():
    """Parse les arguments pour l'optimisation des paramètres VRP."""
    parser = argparse.ArgumentParser(description='Optimiseur de paramètres pour VRP génétique')
    
    # Paramètres de configuration du graphe
    parser.add_argument('--num_cities', type=int, default=20,
                        help='Nombre de villes dans le graphe')
    parser.add_argument('--num_vehicles', type=int, default=3,
                        help='Nombre de véhicules')
    parser.add_argument('--dependency_ratio', type=float, default=0.2,
                        help='Ratio de dépendances entre villes')
    parser.add_argument('--blocking_ratio', type=float, default=0.1,
                        help='Ratio de routes bloquées')
    parser.add_argument('--time_windows', action='store_true',
                        help='Ajouter des fenêtres temporelles')
    
    # Paramètres de l'algorithme génétique fixes
    parser.add_argument('--population_size', type=int, default=500,
                        help='Taille de population fixe')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Nombre maximum d\'itérations fixe')
    
    # Paramètres d'optimisation
    parser.add_argument('--output_dir', type=str, default='vrp_optimization_results',
                        help='Répertoire de sortie pour les résultats')
    parser.add_argument('--runs_per_config', type=int, default=3,
                        help='Nombre d\'exécutions par configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour la reproductibilité')
    
    # Optimisation des mutations
    parser.add_argument('--mutation_rates', type=str, default='0.05,0.1,0.15,0.2,0.25,0.3',
                        help='Taux de mutation à tester, séparés par des virgules')
    
    args = parser.parse_args()
    
    # Convertir les taux de mutation en liste
    args.mutation_rates = [float(x) for x in args.mutation_rates.split(',')]
    
    return args


def run_parameter_optimization(args, mutation_rates):
    """
    Exécute l'optimisation des paramètres en testant différents taux de mutation.
    
    Args:
        args: Arguments de ligne de commande
        mutation_rates: Liste des taux de mutation à tester
        
    Returns:
        DataFrame avec les résultats
    """
    # Créer le répertoire de sortie
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialiser le logger pour collecter les résultats
    logger = ResultsLogger(output_dir=args.output_dir)
    
    # Horodatage pour le nom de fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Préparer le dataframe pour collecter les résultats
    results_df = pd.DataFrame()
    
    # Calculer le nombre total d'exécutions
    total_executions = len(mutation_rates) * args.runs_per_config
    print(f"Lancement de l'optimisation des taux de mutation")
    print(f"Chaque configuration sera exécutée {args.runs_per_config} fois")
    print(f"Total de {total_executions} exécutions prévues")
    
    # Compteur pour suivre l'avancement
    completed_runs = 0
    
    # Tester chaque taux de mutation
    for mutation_rate in mutation_rates:
        print(f"\n--- Test du taux de mutation {mutation_rate} ---")
        
        config_results = []
        
        # Exécuter plusieurs fois avec ce taux de mutation
        for run in range(args.runs_per_config):
            # Générer une graine unique pour cette exécution
            run_seed = args.seed + int(mutation_rate * 10000) + run
            
            run_id = f"mut_{mutation_rate:.3f}_r{run+1}"
            
            try:
                # Générer un graphe VRP
                generator = VRPGraphGenerator(seed=run_seed)
                graph = generator.generate_vrp_graph(
                    num_cities=args.num_cities,
                    num_vehicles=args.num_vehicles,
                    dependency_ratio=args.dependency_ratio,
                    blocking_ratio=args.blocking_ratio,
                    time_windows=args.time_windows
                )
                
                # Si fenêtres temporelles activées, ajouter des temps de service
                if args.time_windows:
                    generator.add_service_times()
                
                # Créer le solveur
                solver = GeneticVRPSolver(
                    graph=graph,
                    vehicles=generator.vehicles,
                    dependencies=graph.graph.get('dependencies', []),
                    depot=generator.depot,
                    seed=run_seed
                )
                
                # Résoudre le problème
                result = solver.solve(
                    population_size=args.population_size,
                    mutation_rate=mutation_rate,
                    max_iterations=args.max_iterations,
                    early_stopping_rounds=50
                )
                
                # Ajouter des informations de configuration
                result['mutation_rate'] = mutation_rate
                
                logger.add_result(result)
                
                # Collecter les métriques importantes
                config_result = {
                    'mutation_rate': mutation_rate,
                    'total_cost': result['total_cost'],
                    'avg_time': result['performance_metrics']['execution_time_sec'],
                    'iterations': result['performance_metrics']['iterations'],
                    'dependencies_respected': int(result['constraints']['dependencies_respected']),
                    'capacity_respected': int(result['constraints']['capacity_respected']),
                    'all_cities_visited': int(result['constraints']['all_cities_visited']),
                    'vehicles_used': result['fleet_metrics']['total_vehicles_used']
                }
                
                config_results.append(config_result)
                
                # Mettre à jour le compteur
                completed_runs += 1
                print(f"  Exécution {run+1}/{args.runs_per_config} terminée. "
                      f"Progression: {completed_runs}/{total_executions} "
                      f"({completed_runs/total_executions*100:.1f}%)")
                
            except Exception as e:
                print(f"    ERREUR dans l'exécution {run_id}: {str(e)}")
        
        # Agréger les résultats pour ce taux de mutation
        if config_results:
            # Calculer les moyennes
            agg_result = {
                'mutation_rate': mutation_rate,
                'avg_cost': np.mean([r['total_cost'] for r in config_results]),
                'std_cost': np.std([r['total_cost'] for r in config_results]),
                'avg_time': np.mean([r['avg_time'] for r in config_results]),
                'avg_iterations': np.mean([r['iterations'] for r in config_results]),
                'dependencies_success_rate': np.mean([r['dependencies_respected'] for r in config_results]) * 100,
                'capacity_success_rate': np.mean([r['capacity_respected'] for r in config_results]) * 100,
                'visits_success_rate': np.mean([r['all_cities_visited'] for r in config_results]) * 100,
                'avg_vehicles_used': np.mean([r['vehicles_used'] for r in config_results])
            }
            
            # Calculer un score composite (à ajuster selon vos priorités)
            agg_result['score'] = (1000 / (agg_result['avg_cost'] + 1)) * \
                                  (agg_result['visits_success_rate'] / 100) * \
                                  (agg_result['dependencies_success_rate'] / 100) * \
                                  (1 / (agg_result['avg_time'] + 1))
            
            results_df = pd.concat([results_df, pd.DataFrame([agg_result])], ignore_index=True)
    
    # Sauvegarder les résultats complets avec le logger
    logger.save_to_json(filename=f"vrp_optimization_full_{timestamp}.json")
    
    # Sauvegarder les résultats agrégés
    csv_path = os.path.join(args.output_dir, f"vrp_optimization_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    
    return results_df


def visualize_optimization_results(results_df, output_dir):
    """
    Génère des visualisations pour comprendre l'impact des différents taux de mutation.
    
    Args:
        results_df: DataFrame contenant les résultats
        output_dir: Répertoire pour sauvegarder les visualisations
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # 1. Impact du taux de mutation sur le coût
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='mutation_rate', y='avg_cost', data=results_df)
    plt.title('Coût moyen par taux de mutation')
    plt.xlabel('Taux de mutation')
    plt.ylabel('Coût moyen')
    plt.xticks(rotation=45)
    
    # 2. Impact du taux de mutation sur le temps d'exécution
    plt.subplot(2, 2, 2)
    sns.barplot(x='mutation_rate', y='avg_time', data=results_df)
    plt.title('Temps moyen par taux de mutation')
    plt.xlabel('Taux de mutation')
    plt.ylabel('Temps moyen (s)')
    plt.xticks(rotation=45)
    
    # 3. Impact du taux de mutation sur le taux de succès des contraintes
    plt.subplot(2, 2, 3)
    results_long = results_df.melt(
        id_vars=['mutation_rate'], 
        value_vars=['dependencies_success_rate', 'capacity_success_rate', 'visits_success_rate'],
        var_name='Constraint', 
        value_name='Success Rate'
    )
    
    sns.barplot(x='mutation_rate', y='Success Rate', hue='Constraint', data=results_long)
    plt.title('Taux de succès des contraintes par taux de mutation')
    plt.xlabel('Taux de mutation')
    plt.ylabel('Taux de succès (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Contrainte', loc='lower right', bbox_to_anchor=(1.25, 0.5))
    
    # 4. Score composite en fonction du taux de mutation
    plt.subplot(2, 2, 4)
    sns.barplot(x='mutation_rate', y='score', data=results_df)
    plt.title('Score composite par taux de mutation')
    plt.xlabel('Taux de mutation')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mutation_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return viz_dir


def find_optimal_mutation_rate(results_df):
    """
    Identifie le taux de mutation optimal.
    
    Args:
        results_df: DataFrame contenant les résultats
        
    Returns:
        Dictionnaire avec les paramètres optimaux
    """
    # Filtrer les configurations avec un bon taux de solutions valides
    valid_threshold = 80  # 80% de solutions valides
    valid_configs = results_df[
        (results_df['visits_success_rate'] >= valid_threshold) & 
        (results_df['dependencies_success_rate'] >= valid_threshold) & 
        (results_df['capacity_success_rate'] >= valid_threshold)
    ]
    
    if valid_configs.empty:
        # Si aucune configuration ne répond au critère, abaisser le seuil
        valid_threshold = 50
        valid_configs = results_df[
            (results_df['visits_success_rate'] >= valid_threshold) | 
            (results_df['dependencies_success_rate'] >= valid_threshold) | 
            (results_df['capacity_success_rate'] >= valid_threshold)
        ]
    
    if valid_configs.empty:
        # Si toujours aucune configuration, prendre toutes les configurations
        valid_configs = results_df
    
    # Trouver la configuration avec le meilleur score
    best_config = valid_configs.loc[valid_configs['score'].idxmax()]
    
    # Extraire les paramètres optimaux
    optimal_params = {
        'mutation_rate': best_config['mutation_rate'],
        'metrics': {
            'avg_cost': best_config['avg_cost'],
            'std_cost': best_config['std_cost'],
            'avg_time': best_config['avg_time'],
            'dependencies_success_rate': best_config['dependencies_success_rate'],
            'capacity_success_rate': best_config['capacity_success_rate'],
            'visits_success_rate': best_config['visits_success_rate'],
            'score': best_config['score']
        }
    }
    
    # Trouver des alternatives intéressantes
    optimal_params['alternatives'] = {}
    
    # 1. Configuration avec le coût le plus bas
    lowest_cost_config = valid_configs.loc[valid_configs['avg_cost'].idxmin()]
    optimal_params['alternatives']['lowest_cost'] = {
        'mutation_rate': lowest_cost_config['mutation_rate'],
        'avg_cost': lowest_cost_config['avg_cost'],
        'visits_success_rate': lowest_cost_config['visits_success_rate']
    }
    
    # 2. Configuration la plus rapide
    fastest_config = valid_configs.loc[valid_configs['avg_time'].idxmin()]
    optimal_params['alternatives']['fastest'] = {
        'mutation_rate': fastest_config['mutation_rate'],
        'avg_time': fastest_config['avg_time'],
        'avg_cost': fastest_config['avg_cost']
    }
    
    # 3. Configuration avec le meilleur taux de succès des contraintes
    best_constraints_config = valid_configs.loc[
        (valid_configs['dependencies_success_rate'] + 
         valid_configs['capacity_success_rate'] + 
         valid_configs['visits_success_rate']).idxmax()
    ]
    optimal_params['alternatives']['best_constraints'] = {
        'mutation_rate': best_constraints_config['mutation_rate'],
        'dependencies_success_rate': best_constraints_config['dependencies_success_rate'],
        'capacity_success_rate': best_constraints_config['capacity_success_rate'],
        'visits_success_rate': best_constraints_config['visits_success_rate']
    }
    
    return optimal_params


def main():
    """Fonction principale pour l'optimisation des paramètres VRP."""
    # Analyser les arguments
    args = parse_optimizer_args()
    
    print(f"\n===== Optimisation des paramètres de l'algorithme génétique VRP =====")
    print(f"Graphe de {args.num_cities} villes, {args.num_vehicles} véhicules")
    print(f"Dépendances: {args.dependency_ratio}, Blocages: {args.blocking_ratio}")
    print(f"Taux de mutation testés: {args.mutation_rates}")
    print(f"Taille de population fixe: {args.population_size}")
    print(f"Nombre max d'itérations: {args.max_iterations}")
    
    if args.time_windows:
        print("Fenêtres temporelles activées")
    
    # Exécuter l'optimisation
    results_df = run_parameter_optimization(args, args.mutation_rates)
    
    # Visualiser les résultats
    viz_dir = visualize_optimization_results(results_df, args.output_dir)
    
    # Trouver les paramètres optimaux
    optimal_params = find_optimal_mutation_rate(results_df)
    
    # Sauvegarder les paramètres optimaux
    optimal_file = os.path.join(args.output_dir, "optimal_mutation_rate.json")
    with open(optimal_file, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    # Afficher un résumé
    print("\n=== RÉSULTATS DE L'OPTIMISATION ===")
    print(f"Résultats détaillés sauvegardés dans: {args.output_dir}")
    print(f"Visualisations disponibles dans: {viz_dir}")
    print(f"Paramètres optimaux sauvegardés dans: {optimal_file}")
    
    print("\n=== PARAMÈTRE DE MUTATION OPTIMAL ===")
    print(f"Taux de mutation: {optimal_params['mutation_rate']}")
    print(f"Score: {optimal_params['metrics']['score']:.2f}")
    print(f"Coût moyen: {optimal_params['metrics']['avg_cost']:.2f}")
    print(f"Temps moyen: {optimal_params['metrics']['avg_time']:.2f}s")
    
    print("\n=== TAUX DE SUCCÈS DES CONTRAINTES ===")
    print(f"Dépendances: {optimal_params['metrics']['dependencies_success_rate']:.1f}%")
    print(f"Capacité: {optimal_params['metrics']['capacity_success_rate']:.1f}%")
    print(f"Visites de toutes les villes: {optimal_params['metrics']['visits_success_rate']:.1f}%")
    
    print("\n=== ALTERNATIVES INTÉRESSANTES ===")
    
    if 'lowest_cost' in optimal_params.get('alternatives', {}):
        lowest_cost = optimal_params['alternatives']['lowest_cost']
        print(f"Configuration avec le coût le plus bas: Mutation {lowest_cost['mutation_rate']} "
              f"(coût: {lowest_cost['avg_cost']:.2f}, visites: {lowest_cost['visits_success_rate']:.1f}%)")
    
    if 'fastest' in optimal_params.get('alternatives', {}):
        fastest = optimal_params['alternatives']['fastest']
        print(f"Configuration la plus rapide: Mutation {fastest['mutation_rate']} "
              f"(temps: {fastest['avg_time']:.2f}s, coût: {fastest['avg_cost']:.2f})")
    
    if 'best_constraints' in optimal_params.get('alternatives', {}):
        best_constraints = optimal_params['alternatives']['best_constraints']
        print(f"Configuration avec le meilleur respect des contraintes: Mutation {best_constraints['mutation_rate']} "
              f"(dépendances: {best_constraints['dependencies_success_rate']:.1f}%, "
              f"capacité: {best_constraints['capacity_success_rate']:.1f}%, "
              f"visites: {best_constraints['visits_success_rate']:.1f}%)")
    
    # Afficher la commande pour utiliser le taux de mutation optimal
    print("\nPour utiliser ce taux de mutation optimal, ajoutez à votre commande :")
    print(f"--mutation_rate {optimal_params['mutation_rate']}")


if __name__ == "__main__":
    main()