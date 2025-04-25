"""
parameter_optimizer.py

Script pour optimiser automatiquement les paramètres de l'algorithme génétique
en effectuant une recherche par grille sur un panel de graphes variés
et en visualisant les résultats pour identifier les configurations les plus robustes.
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

# Importer les modules existants
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_generator import TSPGraphGenerator
from genetic_solver import GeneticTSPSolver
from results_logger import ResultsLogger
from runner import run_single_experiment


def parse_optimizer_args():
    """Parse les arguments pour l'optimisation des paramètres."""
    parser = argparse.ArgumentParser(description='Optimiseur de paramètres pour TSP génétique')
    
    # Mode d'optimisation
    parser.add_argument('--mode', type=str, choices=['single_graph', 'multi_graph', 'robust'], 
                       default='single_graph',
                       help='Mode d\'optimisation: un seul graphe, multiple graphes ou recherche robuste')
    
    # Paramètres du problème pour un graphe spécifique
    parser.add_argument('--num_cities', type=int, default=20,
                        help='Nombre de villes dans le graphe (pour mode single_graph)')
    parser.add_argument('--dependency_ratio', type=float, default=0.2,
                        help='Ratio de dépendances entre villes (pour mode single_graph)')
    parser.add_argument('--blocking_ratio', type=float, default=0.1,
                        help='Ratio de routes bloquées (pour mode single_graph)')
    parser.add_argument('--vehicle_capacity', type=int, default=1000,
                        help='Capacité du véhicule (pour mode single_graph)')
    
    # Paramètres pour la génération de multiples graphes
    parser.add_argument('--city_sizes', type=str, default='10,20,30,50',
                        help='Tailles de villes à tester, séparées par des virgules')
    parser.add_argument('--dependency_ratios', type=str, default='0.1,0.2,0.3',
                        help='Ratios de dépendances à tester, séparés par des virgules')
    parser.add_argument('--blocking_ratios', type=str, default='0.05,0.1,0.15',
                        help='Ratios de routes bloquées à tester, séparés par des virgules')
    parser.add_argument('--vehicle_capacities', type=str, default='500,1000,1500',
                        help='Capacités de véhicule à tester, séparées par des virgules')
    parser.add_argument('--max_test_cases', type=int, default=10,
                        help='Nombre maximal de cas de test en mode robust')
    
    # Paramètres de l'optimisation
    parser.add_argument('--output_dir', type=str, default='optimization_results',
                        help='Répertoire de sortie pour les résultats')
    parser.add_argument('--runs_per_config', type=int, default=3,
                        help='Nombre d\'exécutions par configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour la reproductibilité')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Nombre maximal de configurations à tester (pour accélérer les tests)')
    
    # Options d'optimisation des paramètres de l'algorithme
    parser.add_argument('--optimize_population', action='store_true',
                        help='Optimiser la taille de la population')
    parser.add_argument('--optimize_mutation', action='store_true',
                        help='Optimiser le taux de mutation')
    parser.add_argument('--optimize_iterations', action='store_true',
                        help='Optimiser le nombre maximal d\'itérations')
    parser.add_argument('--optimize_early_stopping', action='store_true',
                        help='Optimiser le seuil d\'arrêt anticipé')
    parser.add_argument('--optimize_all', action='store_true',
                        help='Optimiser tous les paramètres')
    
    args = parser.parse_args()
    
    # Si optimize_all est activé, activer toutes les options d'optimisation
    if args.optimize_all:
        args.optimize_population = True
        args.optimize_mutation = True
        args.optimize_iterations = True
        args.optimize_early_stopping = True
    
    # Convertir les listes de paramètres de chaînes en listes numériques
    args.city_sizes = [int(x) for x in args.city_sizes.split(',')]
    args.dependency_ratios = [float(x) for x in args.dependency_ratios.split(',')]
    args.blocking_ratios = [float(x) for x in args.blocking_ratios.split(',')]
    args.vehicle_capacities = [int(x) for x in args.vehicle_capacities.split(',')]
    
    return args


def create_parameter_grid(args):
    """
    Crée une grille de paramètres à tester en fonction des options d'optimisation.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        Liste de dictionnaires de configurations à tester
    """
    # Valeurs par défaut
    population_sizes = [100]
    mutation_rates = [0.1]
    max_iterations = [500]
    early_stopping_values = [50]
    
    # Ajouter des valeurs en fonction des options d'optimisation
    if args.optimize_population:
        population_sizes = [50, 100, 200, 300, 500]
    
    if args.optimize_mutation:
        mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    if args.optimize_iterations:
        max_iterations = [200, 500, 1000, 2000]
    
    if args.optimize_early_stopping:
        early_stopping_values = [20, 50, 100, 200]
    
    # Créer toutes les combinaisons de paramètres
    param_grid = list(itertools.product(
        population_sizes,
        mutation_rates,
        max_iterations,
        early_stopping_values
    ))
    
    # Convertir en liste de dictionnaires
    config_list = []
    for pop_size, mut_rate, max_iter, early_stop in param_grid:
        config = {
            'population_size': pop_size,
            'mutation_rate': mut_rate,
            'max_iterations': max_iter,
            'early_stopping': early_stop
        }
        config_list.append(config)
    
    return config_list


def generate_test_cases(args):
    """
    Génère un ensemble de cas de test (graphes) selon le mode d'optimisation.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        Liste de configurations de graphes à tester
    """
    if args.mode == 'single_graph':
        # Un seul graphe avec les paramètres spécifiés
        return [{
            'num_cities': args.num_cities,
            'dependency_ratio': args.dependency_ratio,
            'blocking_ratio': args.blocking_ratio,
            'vehicle_capacity': args.vehicle_capacity,
            'geographical': True,
            'seed': args.seed
        }]
    
    else:  # 'multi_graph' ou 'robust'
        # Plusieurs graphes avec différentes configurations
        test_cases = []
        for num_cities in args.city_sizes:
            for dep_ratio in args.dependency_ratios:
                for block_ratio in args.blocking_ratios:
                    for vehicle_cap in args.vehicle_capacities:
                        test_cases.append({
                            'num_cities': num_cities,
                            'dependency_ratio': dep_ratio,
                            'blocking_ratio': block_ratio,
                            'vehicle_capacity': vehicle_cap,
                            'geographical': True,
                            'seed': args.seed
                        })
        
        # Si max_test_cases est spécifié, limiter le nombre de cas
        if args.max_test_cases and len(test_cases) > args.max_test_cases:
            if args.mode == 'robust':
                # Pour le mode robuste, prendre un sous-ensemble varié
                # En sélectionnant intelligemment des cas représentatifs
                selected_cases = []
                
                # Sélectionner au moins un cas de chaque taille de ville
                for size in args.city_sizes:
                    size_cases = [case for case in test_cases if case['num_cities'] == size]
                    if size_cases:
                        # Prendre un cas aléatoire pour cette taille
                        random.seed(args.seed)
                        selected_cases.append(random.choice(size_cases))
                
                # Compléter avec des cas aléatoires
                remaining = args.max_test_cases - len(selected_cases)
                if remaining > 0:
                    remaining_cases = [case for case in test_cases if case not in selected_cases]
                    random.seed(args.seed)
                    random.shuffle(remaining_cases)
                    selected_cases.extend(remaining_cases[:remaining])
                
                test_cases = selected_cases
            else:
                # Pour mode multi_graph, simplement prendre un échantillon aléatoire
                random.seed(args.seed)
                random.shuffle(test_cases)
                test_cases = test_cases[:args.max_test_cases]
        
        return test_cases


def run_parameter_optimization(args, config_list):
    """
    Exécute l'optimisation des paramètres en testant chaque configuration.
    
    Args:
        args: Arguments de ligne de commande
        config_list: Liste des configurations à tester
        
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
    
    # Générer les cas de test (graphes)
    test_cases = generate_test_cases(args)
    print(f"Nombre de graphes de test générés: {len(test_cases)}")
    
    # Limiter le nombre de configurations à tester si spécifié
    if args.max_configs and len(config_list) > args.max_configs:
        random.seed(args.seed)
        random.shuffle(config_list)
        config_list = config_list[:args.max_configs]
        print(f"Nombre de configurations limité à {args.max_configs}")
    
    # Préparer le dataframe pour collecter les résultats
    results_df = pd.DataFrame()
    
    total_executions = len(config_list) * len(test_cases) * args.runs_per_config
    print(f"Lancement de l'optimisation avec {len(config_list)} configurations sur {len(test_cases)} graphes")
    print(f"Chaque combinaison sera exécutée {args.runs_per_config} fois")
    print(f"Total de {total_executions} exécutions prévues")
    
    # Compteur pour suivre l'avancement
    completed_runs = 0
    
    # Tester chaque configuration sur chaque graphe
    for config_idx, config in enumerate(config_list):
        print(f"\n--- Configuration {config_idx+1}/{len(config_list)} ---")
        print(f"Population: {config['population_size']}, Mutation: {config['mutation_rate']}")
        print(f"Itérations max: {config['max_iterations']}, Arrêt anticipé: {config['early_stopping']}")
        
        config_results = []
        
        for case_idx, case in enumerate(test_cases):
            print(f"\n  Cas de test {case_idx+1}/{len(test_cases)}")
            print(f"  Villes: {case['num_cities']}, Dépendances: {case['dependency_ratio']}, "
                  f"Blocages: {case['blocking_ratio']}, Capacité: {case['vehicle_capacity']}")
            
            # Paramètres de base pour ce cas de test
            base_args = argparse.Namespace(
                num_cities=case['num_cities'],
                dependency_ratio=case['dependency_ratio'],
                blocking_ratio=case['blocking_ratio'],
                vehicle_capacity=case['vehicle_capacity'],
                geographical=case['geographical'],
                visualize=False,
                save_graph=False,
                min_load=10,
                max_load=100
            )
            
            # Ajouter les paramètres de l'algorithme
            args_copy = argparse.Namespace(**vars(base_args))
            args_copy.population_size = config['population_size']
            args_copy.mutation_rate = config['mutation_rate']
            args_copy.max_iterations = config['max_iterations']
            args_copy.early_stopping = config['early_stopping']
            
            # Variables pour calculer les métriques moyennes
            total_cost_sum = 0
            execution_time_sum = 0
            iterations_sum = 0
            dependencies_respected_count = 0
            capacity_respected_count = 0
            
            # Exécuter plusieurs fois avec cette configuration et ce cas de test
            for run in range(args.runs_per_config):
                # Générer une graine unique pour cette exécution
                run_seed = args.seed + config_idx * 10000 + case_idx * 100 + run
                
                run_id = f"opt_c{config_idx+1}_g{case_idx+1}_r{run+1}"
                
                try:
                    # Exécuter l'expérimentation
                    result = run_single_experiment(args_copy, seed=run_seed, run_id=run_id)
                    logger.add_result(result)
                    
                    # Collecter les métriques
                    total_cost_sum += result['total_cost']
                    execution_time_sum += result['performance_metrics']['execution_time_sec']
                    iterations_sum += result['performance_metrics']['iterations']
                    dependencies_respected_count += int(result['constraints']['dependencies_respected'])
                    capacity_respected_count += int(result['constraints']['capacity_respected'])
                    
                    # Mettre à jour le compteur
                    completed_runs += 1
                    print(f"    Exécution {run+1}/{args.runs_per_config} terminée. "
                          f"Progression: {completed_runs}/{total_executions} "
                          f"({completed_runs/total_executions*100:.1f}%)")
                    
                except Exception as e:
                    print(f"    ERREUR dans l'exécution {run_id}: {str(e)}")
            
            # Calculer les moyennes pour ce cas de test
            num_runs = args.runs_per_config
            avg_cost = total_cost_sum / num_runs
            avg_time = execution_time_sum / num_runs
            avg_iterations = iterations_sum / num_runs
            dependency_success_rate = dependencies_respected_count / num_runs * 100
            capacity_success_rate = capacity_respected_count / num_runs * 100
            
            # Calculer le taux de solutions valides et le score
            valid_solutions_rate = (dependencies_respected_count / num_runs) * (capacity_respected_count / num_runs) * 100
            
            # Score personnalisé (à adapter selon vos priorités)
            # Plus le score est élevé, meilleure est la configuration
            score = 1000 / (avg_cost + 1) * (valid_solutions_rate / 100) * (1 + 1 / (avg_time + 1))
            
            # Ajouter les résultats de ce cas de test
            case_result = {
                'population_size': config['population_size'],
                'mutation_rate': config['mutation_rate'],
                'max_iterations': config['max_iterations'],
                'early_stopping': config['early_stopping'],
                'num_cities': case['num_cities'],
                'dependency_ratio': case['dependency_ratio'],
                'blocking_ratio': case['blocking_ratio'],
                'vehicle_capacity': case['vehicle_capacity'],
                'avg_cost': avg_cost,
                'avg_time': avg_time,
                'avg_iterations': avg_iterations,
                'dependency_success_rate': dependency_success_rate,
                'capacity_success_rate': capacity_success_rate,
                'valid_solutions_rate': valid_solutions_rate,
                'score': score
            }
            
            config_results.append(case_result)
        
        # Pour le mode robuste, agréger les résultats sur tous les cas de test
        if args.mode == 'robust':
            # Calculer les moyennes et écarts-types sur tous les cas de test
            avg_scores = [r['score'] for r in config_results]
            avg_costs = [r['avg_cost'] for r in config_results]
            valid_rates = [r['valid_solutions_rate'] for r in config_results]
            
            robust_result = {
                'population_size': config['population_size'],
                'mutation_rate': config['mutation_rate'],
                'max_iterations': config['max_iterations'],
                'early_stopping': config['early_stopping'],
                'avg_score': np.mean(avg_scores),
                'std_score': np.std(avg_scores),
                'min_score': min(avg_scores),
                'avg_cost': np.mean(avg_costs),
                'std_cost': np.std(avg_costs),
                'avg_valid_rate': np.mean(valid_rates),
                'min_valid_rate': min(valid_rates),
                # Robustesse = score moyen - écart-type (favorise les configurations stables)
                'robustness': np.mean(avg_scores) - np.std(avg_scores)
            }
            
            # Ajouter à notre dataframe
            results_df = pd.concat([results_df, pd.DataFrame([robust_result])], ignore_index=True)
        else:
            # Ajouter tous les résultats individuels
            results_df = pd.concat([results_df, pd.DataFrame(config_results)], ignore_index=True)
        
        # Sauvegarder les résultats partiels après chaque configuration
        csv_path = os.path.join(args.output_dir, f"optimization_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
    
    # Sauvegarder également les résultats complets avec le logger
    logger.save_to_json(filename=f"optimization_full_{timestamp}.json")
    
    return results_df


def visualize_optimization_results(results_df, output_dir, mode='single_graph'):
    """
    Génère des visualisations pour comprendre l'impact des différents paramètres.
    
    Args:
        results_df: DataFrame contenant les résultats
        output_dir: Répertoire pour sauvegarder les visualisations
        mode: Mode d'optimisation ('single_graph', 'multi_graph', ou 'robust')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Adapter les visualisations selon le mode
    if mode == 'robust':
        # Pour le mode robust, utiliser des visualisations spécifiques à la robustesse
        visualize_robust_results(results_df, viz_dir)
    else:
        # Pour les modes single_graph et multi_graph, utiliser des visualisations standard
        visualize_standard_results(results_df, viz_dir, mode)
    
    return viz_dir


def visualize_standard_results(results_df, viz_dir, mode):
    """
    Génère des visualisations standard pour les modes single_graph et multi_graph.
    
    Args:
        results_df: DataFrame contenant les résultats
        viz_dir: Répertoire pour sauvegarder les visualisations
        mode: Mode d'optimisation ('single_graph' ou 'multi_graph')
    """
    # 1. Impact de la taille de population et du taux de mutation sur le coût
    plt.figure(figsize=(12, 10))
    
    # Créer un pivot pour l'analyse
    pivot_data = results_df.pivot_table(
        values='avg_cost',
        index='population_size',
        columns='mutation_rate',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu_r")
    plt.title('Impact de la taille de population et du taux de mutation sur le coût moyen')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'population_mutation_cost.png'), dpi=300)
    plt.close()
    
    # 2. Impact de la taille de population et du taux de mutation sur le taux de solutions valides
    plt.figure(figsize=(12, 10))
    
    pivot_data = results_df.pivot_table(
        values='valid_solutions_rate',
        index='population_size',
        columns='mutation_rate',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100)
    plt.title('Impact de la taille de population et du taux de mutation sur le taux de solutions valides (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'population_mutation_valid.png'), dpi=300)
    plt.close()
    
    # 3. Impact du nombre d'itérations et de l'arrêt anticipé sur le temps d'exécution
    plt.figure(figsize=(12, 10))
    
    pivot_data = results_df.pivot_table(
        values='avg_time',
        index='max_iterations',
        columns='early_stopping',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title('Impact du nombre d\'itérations et de l\'arrêt anticipé sur le temps d\'exécution moyen (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'iterations_stopping_time.png'), dpi=300)
    plt.close()
    
    # 4. Graphique de comparaison des meilleures configurations basées sur le score
    plt.figure(figsize=(14, 8))
    
    # Trouver les 10 meilleures configurations
    top_configs = results_df.sort_values('score', ascending=False).head(10)
    
    # Créer des étiquettes pour les configurations
    config_labels = [f"P{row['population_size']}_M{row['mutation_rate']}_I{row['max_iterations']}_S{row['early_stopping']}" 
                     for _, row in top_configs.iterrows()]
    
    # Faire un barplot du score
    plt.subplot(1, 2, 1)
    sns.barplot(x=top_configs['score'], y=config_labels)
    plt.title('Score des 10 meilleures configurations')
    plt.xlabel('Score (plus élevé = meilleur)')
    
    # Faire un barplot du coût moyen
    plt.subplot(1, 2, 2)
    sns.barplot(x=top_configs['avg_cost'], y=config_labels)
    plt.title('Coût moyen des 10 meilleures configurations')
    plt.xlabel('Coût moyen (plus bas = meilleur)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_configurations.png'), dpi=300)
    plt.close()
    
    # 5. Impact de chaque paramètre sur le score (graphiques de violon)
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    sns.violinplot(x='population_size', y='score', data=results_df)
    plt.title('Impact de la taille de population sur le score')
    
    plt.subplot(2, 2, 2)
    sns.violinplot(x='mutation_rate', y='score', data=results_df)
    plt.title('Impact du taux de mutation sur le score')
    
    plt.subplot(2, 2, 3)
    sns.violinplot(x='max_iterations', y='score', data=results_df)
    plt.title('Impact du nombre max d\'itérations sur le score')
    
    plt.subplot(2, 2, 4)
    sns.violinplot(x='early_stopping', y='score', data=results_df)
    plt.title('Impact du seuil d\'arrêt anticipé sur le score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'parameter_impact_score.png'), dpi=300)
    plt.close()
    
    # 6. Relation entre coût moyen et temps d'exécution (scatter plot)
    plt.figure(figsize=(12, 8))
    
    # Utiliser différentes tailles de points selon le taux de solutions valides
    sns.scatterplot(
        x='avg_cost', 
        y='avg_time', 
        size='valid_solutions_rate',
        hue='population_size',
        data=results_df,
        sizes=(20, 200),
        alpha=0.7
    )
    
    plt.title('Relation entre coût moyen et temps d\'exécution')
    plt.xlabel('Coût moyen')
    plt.ylabel('Temps d\'exécution moyen (s)')
    plt.legend(title='Taille pop.')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cost_vs_time.png'), dpi=300)
    plt.close()
    
    # Si nous sommes en mode multi_graph, ajouter des visualisations supplémentaires
    if mode == 'multi_graph' and 'num_cities' in results_df.columns:
        # 7. Impact de la taille du problème sur le score par configuration
        plt.figure(figsize=(14, 8))
        
        # Visualiser l'impact de la taille de ville sur le score pour différentes configurations
        sns.lineplot(
            data=results_df,
            x='num_cities',
            y='score',
            hue='population_size',
            style='mutation_rate',
            markers=True,
            dashes=False
        )
        
        plt.title('Impact de la taille du problème sur le score')
        plt.xlabel('Nombre de villes')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'problem_size_impact.png'), dpi=300)
        plt.close()
        
        # 8. Heatmap de l'impact combiné du ratio de dépendance et de blocage
        plt.figure(figsize=(12, 9))
        
        if len(results_df['dependency_ratio'].unique()) > 1 and len(results_df['blocking_ratio'].unique()) > 1:
            pivot_constraints = results_df.pivot_table(
                values='valid_solutions_rate',
                index='dependency_ratio',
                columns='blocking_ratio',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_constraints, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title('Impact combiné des ratios de dépendance et de blocage sur le taux de solutions valides')
            plt.xlabel('Ratio de blocage')
            plt.ylabel('Ratio de dépendance')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'constraint_impact.png'), dpi=300)
            plt.close()


def visualize_robust_results(results_df, viz_dir):
    """
    Génère des visualisations spécifiques pour le mode d'optimisation robuste.
    
    Args:
        results_df: DataFrame contenant les résultats agrégés
        viz_dir: Répertoire pour sauvegarder les visualisations
    """
    # 1. Comparaison des configurations les plus robustes
    plt.figure(figsize=(14, 10))
    
    # Trouver les 10 configurations les plus robustes
    top_robust = results_df.sort_values('robustness', ascending=False).head(10)
    
    # Créer des étiquettes pour les configurations
    config_labels = [f"P{row['population_size']}_M{row['mutation_rate']}_I{row['max_iterations']}_S{row['early_stopping']}" 
                     for _, row in top_robust.iterrows()]
    
    # Barplot de la robustesse
    plt.subplot(2, 2, 1)
    sns.barplot(x=top_robust['robustness'], y=config_labels)
    plt.title('Robustesse des configurations')
    plt.xlabel('Robustesse (score moyen - écart-type)')
    
    # Barplot du score moyen
    plt.subplot(2, 2, 2)
    sns.barplot(x=top_robust['avg_score'], y=config_labels)
    plt.title('Score moyen')
    plt.xlabel('Score moyen')
    
    # Barplot de l'écart-type du score
    plt.subplot(2, 2, 3)
    sns.barplot(x=top_robust['std_score'], y=config_labels)
    plt.title('Écart-type du score')
    plt.xlabel('Écart-type (plus bas = plus stable)')
    
    # Barplot du taux moyen de solutions valides
    plt.subplot(2, 2, 4)
    sns.barplot(x=top_robust['avg_valid_rate'], y=config_labels)
    plt.title('Taux moyen de solutions valides')
    plt.xlabel('% de solutions valides')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'robust_configurations.png'), dpi=300)
    plt.close()
    
    # 2. Trade-off entre score moyen et stabilité
    plt.figure(figsize=(12, 8))
    
    # Scatter plot montrant le compromis entre score moyen et écart-type
    sns.scatterplot(
        x='avg_score',
        y='std_score',
        size='avg_valid_rate',
        hue='population_size',
        data=results_df,
        sizes=(20, 200),
        alpha=0.7
    )
    
    # Ajouter des annotations pour les configurations les plus intéressantes
    for i, row in top_robust.head(5).iterrows():
        plt.annotate(
            f"P{row['population_size']}_M{row['mutation_rate']}",
            xy=(row['avg_score'], row['std_score']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('Compromis entre performance moyenne et stabilité')
    plt.xlabel('Score moyen (plus élevé = meilleur)')
    plt.ylabel('Écart-type (plus bas = plus stable)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'robust_tradeoff.png'), dpi=300)
    plt.close()
    
    # 3. Impact de chaque paramètre sur la robustesse
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='population_size', y='robustness', data=results_df)
    plt.title('Impact de la taille de population sur la robustesse')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='mutation_rate', y='robustness', data=results_df)
    plt.title('Impact du taux de mutation sur la robustesse')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='max_iterations', y='robustness', data=results_df)
    plt.title('Impact du nombre max d\'itérations sur la robustesse')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='early_stopping', y='robustness', data=results_df)
    plt.title('Impact du seuil d\'arrêt anticipé sur la robustesse')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'parameter_impact_robustness.png'), dpi=300)
    plt.close()


def find_optimal_parameters(results_df, mode='single_graph'):
    """
    Identifie les paramètres optimaux en fonction des résultats.
    
    Args:
        results_df: DataFrame contenant les résultats
        mode: Mode d'optimisation ('single_graph', 'multi_graph', ou 'robust')
        
    Returns:
        Dictionnaire contenant les paramètres optimaux et leur justification
    """
    # Adapter la recherche selon le mode
    if mode == 'robust':
        return find_robust_parameters(results_df)
    
    # Filtrer les configurations avec un bon taux de solutions valides
    valid_threshold = 80  # 80% de solutions valides
    valid_configs = results_df[results_df['valid_solutions_rate'] >= valid_threshold]
    
    if valid_configs.empty:
        # Si aucune configuration ne répond au critère, abaisser le seuil
        valid_threshold = 50
        valid_configs = results_df[results_df['valid_solutions_rate'] >= valid_threshold]
    
    if valid_configs.empty:
        # Si toujours aucune configuration, prendre celles avec le meilleur taux
        max_valid_rate = results_df['valid_solutions_rate'].max()
        valid_configs = results_df[results_df['valid_solutions_rate'] == max_valid_rate]
    
    # Parmi les configurations valides, trouver celle avec le meilleur score
    best_config = valid_configs.loc[valid_configs['score'].idxmax()]
    
    # Extraire les paramètres optimaux
    optimal_params = {
        'population_size': int(best_config['population_size']),
        'mutation_rate': best_config['mutation_rate'],
        'max_iterations': int(best_config['max_iterations']),
        'early_stopping': int(best_config['early_stopping']),
        'metrics': {
            'avg_cost': best_config['avg_cost'],
            'avg_time': best_config['avg_time'],
            'avg_iterations': best_config['avg_iterations'],
            'dependency_success_rate': best_config['dependency_success_rate'],
            'capacity_success_rate': best_config['capacity_success_rate'],
            'valid_solutions_rate': best_config['valid_solutions_rate'],
            'score': best_config['score']
        }
    }
    
    # Pour le mode multi_graph, ajouter des informations sur la distribution des résultats
    if mode == 'multi_graph' and 'num_cities' in results_df.columns:
        # Ajouter des statistiques sur les performances par taille de problème
        city_sizes = sorted(results_df['num_cities'].unique())
        size_performance = {}
        
        for size in city_sizes:
            size_data = results_df[
                (results_df['num_cities'] == size) & 
                (results_df['population_size'] == optimal_params['population_size']) &
                (results_df['mutation_rate'] == optimal_params['mutation_rate']) &
                (results_df['max_iterations'] == optimal_params['max_iterations']) &
                (results_df['early_stopping'] == optimal_params['early_stopping'])
            ]
            
            if not size_data.empty:
                size_performance[str(size)] = {
                    'avg_cost': size_data['avg_cost'].mean(),
                    'valid_solutions_rate': size_data['valid_solutions_rate'].mean(),
                    'avg_time': size_data['avg_time'].mean()
                }
        
        optimal_params['size_performance'] = size_performance
    
    # Trouver des alternatives intéressantes
    optimal_params['alternatives'] = {}
    
    # 1. La solution la plus rapide avec un bon ratio de validité
    if not valid_configs.empty:
        fastest_config = valid_configs.loc[valid_configs['avg_time'].idxmin()]
        optimal_params['alternatives']['fastest'] = {
            'population_size': int(fastest_config['population_size']),
            'mutation_rate': fastest_config['mutation_rate'],
            'max_iterations': int(fastest_config['max_iterations']),
            'early_stopping': int(fastest_config['early_stopping']),
            'avg_time': fastest_config['avg_time'],
            'score': fastest_config['score']
        }
    
    # 2. La solution avec le meilleur coût (même si elle est plus lente)
    best_cost_config = results_df.loc[results_df['avg_cost'].idxmin()]
    optimal_params['alternatives']['best_cost'] = {
        'population_size': int(best_cost_config['population_size']),
        'mutation_rate': best_cost_config['mutation_rate'],
        'max_iterations': int(best_cost_config['max_iterations']),
        'early_stopping': int(best_cost_config['early_stopping']),
        'avg_cost': best_cost_config['avg_cost'],
        'valid_solutions_rate': best_cost_config['valid_solutions_rate'],
        'score': best_cost_config['score']
    }
    
    return optimal_params


def find_robust_parameters(results_df):
    """
    Identifie les paramètres les plus robustes en fonction des résultats agrégés.
    
    Args:
        results_df: DataFrame contenant les résultats agrégés
        
    Returns:
        Dictionnaire contenant les paramètres les plus robustes
    """
    # Filtrer les configurations avec un bon taux de solutions valides
    valid_threshold = 70  # Seuil légèrement plus bas pour le mode robuste
    valid_configs = results_df[results_df['avg_valid_rate'] >= valid_threshold]
    
    if valid_configs.empty:
        # Si aucune configuration ne répond au critère, abaisser le seuil
        valid_threshold = 50
        valid_configs = results_df[results_df['avg_valid_rate'] >= valid_threshold]
    
    if valid_configs.empty:
        # Si toujours aucune configuration, prendre toutes les configurations
        valid_configs = results_df
    
    # Parmi les configurations valides, trouver celle avec la meilleure robustesse
    best_config = valid_configs.loc[valid_configs['robustness'].idxmax()]
    
    # Extraire les paramètres optimaux
    robust_params = {
        'population_size': int(best_config['population_size']),
        'mutation_rate': best_config['mutation_rate'],
        'max_iterations': int(best_config['max_iterations']),
        'early_stopping': int(best_config['early_stopping']),
        'metrics': {
            'avg_score': best_config['avg_score'],
            'std_score': best_config['std_score'],
            'min_score': best_config['min_score'],
            'avg_cost': best_config['avg_cost'],
            'std_cost': best_config['std_cost'],
            'avg_valid_rate': best_config['avg_valid_rate'],
            'min_valid_rate': best_config['min_valid_rate'],
            'robustness': best_config['robustness']
        }
    }
    
    # Trouver des alternatives intéressantes
    robust_params['alternatives'] = {}
    
    # 1. La configuration avec le meilleur score moyen
    best_avg_config = valid_configs.loc[valid_configs['avg_score'].idxmax()]
    robust_params['alternatives']['best_average'] = {
        'population_size': int(best_avg_config['population_size']),
        'mutation_rate': best_avg_config['mutation_rate'],
        'max_iterations': int(best_avg_config['max_iterations']),
        'early_stopping': int(best_avg_config['early_stopping']),
        'avg_score': best_avg_config['avg_score'],
        'std_score': best_avg_config['std_score'],
        'robustness': best_avg_config['robustness']
    }
    
    # 2. La configuration la plus stable (écart-type minimal)
    most_stable_config = valid_configs.loc[valid_configs['std_score'].idxmin()]
    robust_params['alternatives']['most_stable'] = {
        'population_size': int(most_stable_config['population_size']),
        'mutation_rate': most_stable_config['mutation_rate'],
        'max_iterations': int(most_stable_config['max_iterations']),
        'early_stopping': int(most_stable_config['early_stopping']),
        'avg_score': most_stable_config['avg_score'],
        'std_score': most_stable_config['std_score'],
        'robustness': most_stable_config['robustness']
    }
    
    return robust_params


def main():
    """Fonction principale"""
    # Analyser les arguments
    args = parse_optimizer_args()
    
    print(f"\n===== Optimisation des paramètres de l'algorithme génétique TSP =====")
    print(f"Mode: {args.mode}")
    
    # Créer la grille de paramètres à tester
    config_list = create_parameter_grid(args)
    
    print(f"Nombre de configurations à tester: {len(config_list)}")
    
    if args.mode == 'single_graph':
        print(f"Graphe de {args.num_cities} villes avec dépendances {args.dependency_ratio}, "
              f"blocages {args.blocking_ratio}, capacité {args.vehicle_capacity}")
    else:
        print(f"Test sur plusieurs graphes avec:")
        print(f"  - Tailles de villes: {args.city_sizes}")
        print(f"  - Ratios de dépendances: {args.dependency_ratios}")
        print(f"  - Ratios de blocages: {args.blocking_ratios}")
        print(f"  - Capacités de véhicule: {args.vehicle_capacities}")
    
    # Exécuter l'optimisation
    results_df = run_parameter_optimization(args, config_list)
    
    # Visualiser les résultats
    viz_dir = visualize_optimization_results(results_df, args.output_dir, args.mode)
    
    # Trouver les paramètres optimaux
    optimal_params = find_optimal_parameters(results_df, args.mode)
    
    # Sauvegarder les paramètres optimaux
    optimal_file = os.path.join(args.output_dir, "optimal_parameters.json")
    with open(optimal_file, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    # Afficher un résumé
    print("\n=== RÉSULTATS DE L'OPTIMISATION ===")
    print(f"Nombre de configurations testées: {len(config_list)}")
    print(f"Résultats détaillés sauvegardés dans: {args.output_dir}")
    print(f"Visualisations disponibles dans: {viz_dir}")
    print(f"Paramètres optimaux sauvegardés dans: {optimal_file}")
    
    if args.mode == 'robust':
        print("\n=== PARAMÈTRES LES PLUS ROBUSTES ===")
        print(f"Taille de population: {optimal_params['population_size']}")
        print(f"Taux de mutation: {optimal_params['mutation_rate']}")
        print(f"Nombre max d'itérations: {optimal_params['max_iterations']}")
        print(f"Arrêt anticipé: {optimal_params['early_stopping']}")
        print(f"Robustesse: {optimal_params['metrics']['robustness']:.2f}")
        print(f"Score moyen: {optimal_params['metrics']['avg_score']:.2f}")
        print(f"Écart-type: {optimal_params['metrics']['std_score']:.2f}")
        print(f"Taux moyen de solutions valides: {optimal_params['metrics']['avg_valid_rate']:.1f}%")
    else:
        print("\n=== PARAMÈTRES OPTIMAUX ===")
        print(f"Taille de population: {optimal_params['population_size']}")
        print(f"Taux de mutation: {optimal_params['mutation_rate']}")
        print(f"Nombre max d'itérations: {optimal_params['max_iterations']}")
        print(f"Arrêt anticipé: {optimal_params['early_stopping']}")
        print(f"Score: {optimal_params['metrics']['score']:.2f}")
        print(f"Coût moyen: {optimal_params['metrics']['avg_cost']:.2f}")
        print(f"Taux de solutions valides: {optimal_params['metrics']['valid_solutions_rate']:.1f}%")
    
    print("\n=== ALTERNATIVES INTÉRESSANTES ===")
    
    if args.mode == 'robust':
        if 'best_average' in optimal_params.get('alternatives', {}):
            best_avg = optimal_params['alternatives']['best_average']
            print(f"Config avec le meilleur score moyen: P{best_avg['population_size']}_M{best_avg['mutation_rate']} "
                  f"(score: {best_avg['avg_score']:.2f}, écart-type: {best_avg['std_score']:.2f})")
        
        if 'most_stable' in optimal_params.get('alternatives', {}):
            most_stable = optimal_params['alternatives']['most_stable']
            print(f"Config la plus stable: P{most_stable['population_size']}_M{most_stable['mutation_rate']} "
                  f"(écart-type: {most_stable['std_score']:.2f}, score: {most_stable['avg_score']:.2f})")
    else:
        if 'fastest' in optimal_params.get('alternatives', {}):
            fastest = optimal_params['alternatives']['fastest']
            print(f"Config la plus rapide: P{fastest['population_size']}_M{fastest['mutation_rate']} "
                  f"(temps: {fastest['avg_time']:.2f}s)")
        
        if 'best_cost' in optimal_params.get('alternatives', {}):
            best_cost = optimal_params['alternatives']['best_cost']
            print(f"Config avec le meilleur coût: P{best_cost['population_size']}_M{best_cost['mutation_rate']} "
                  f"(coût: {best_cost['avg_cost']:.2f}, validité: {best_cost['valid_solutions_rate']:.1f}%)")
    
    # Afficher la commande pour utiliser les paramètres optimaux
    print("\nPour utiliser ces paramètres optimaux, exécutez:")
    print(f"python runner.py --population_size {optimal_params['population_size']} "
          f"--mutation_rate {optimal_params['mutation_rate']} "
          f"--max_iterations {optimal_params['max_iterations']} "
          f"--early_stopping {optimal_params['early_stopping']}")
    
    # Si nous avons des performances par taille, donner des recommandations adaptées
    if args.mode == 'multi_graph' and 'size_performance' in optimal_params:
        print("\n=== RECOMMANDATIONS PAR TAILLE DE PROBLÈME ===")
        for size, perf in optimal_params['size_performance'].items():
            print(f"Pour {size} villes: validité {perf['valid_solutions_rate']:.1f}%, "
                  f"coût {perf['avg_cost']:.2f}, temps {perf['avg_time']:.2f}s")


if __name__ == "__main__":
    main()