# Remplacer ces lignes
from graph_generator import TSPGraphGenerator
from genetic_solver import GeneticTSPSolver
from results_logger import ResultsLogger

import os
import argparse
import time
import random
import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union


def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Runner pour le TSP avec contraintes hybrides')
    
    # Configuration du graphe
    parser.add_argument('--num_cities', type=int, default=20, 
                        help='Nombre de villes dans le graphe')
    parser.add_argument('--geographical', action='store_true', 
                        help='Utiliser des distances géographiques plutôt qu\'aléatoires')
    parser.add_argument('--min_load', type=int, default=10, 
                        help='Charge minimale pour une ville')
    parser.add_argument('--max_load', type=int, default=100, 
                        help='Charge maximale pour une ville')
    parser.add_argument('--dependency_ratio', type=float, default=0.2, 
                        help='Ratio de dépendances entre villes')
    parser.add_argument('--blocking_ratio', type=float, default=0.1, 
                        help='Ratio de routes bloquées')
    
    # Configuration de l'algorithme génétique
    parser.add_argument('--population_size', type=int, default=100, 
                        help='Taille de la population pour l\'algorithme génétique')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                        help='Taux de mutation pour l\'algorithme génétique')
    parser.add_argument('--max_iterations', type=int, default=500, 
                        help='Nombre maximum d\'itérations pour l\'algorithme génétique')
    parser.add_argument('--early_stopping', type=int, default=50, 
                        help='Arrêter si pas d\'amélioration pendant N itérations')
    parser.add_argument('--vehicle_capacity', type=int, default=1000, 
                        help='Capacité du véhicule')
    
    # Configuration des expérimentations
    parser.add_argument('--num_runs', type=int, default=1, 
                        help='Nombre d\'exécutions avec des graines différentes')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Répertoire de sortie pour les résultats')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Graine aléatoire pour la reproductibilité (optionnel)')
    parser.add_argument('--save_graph', action='store_true', 
                        help='Sauvegarder le graphe généré au format JSON')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualiser le graphe et la solution')
    
    args = parser.parse_args()
    return args


def visualize_graph_and_solution(graph: nx.Graph, solution_path: List[str], 
                                output_dir: str, run_id: str):
    """
    Visualise le graphe et la solution trouvée.
    
    Args:
        graph: Graphe NetworkX
        solution_path: Chemin solution (liste ordonnée de villes)
        output_dir: Répertoire de sortie pour les images
        run_id: Identifiant de l'exécution
    """
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10))
    
    # Vérifier si le graphe a des positions géographiques
    if all('pos' in graph.nodes[city] for city in graph.nodes):
        pos = nx.get_node_attributes(graph, 'pos')
    else:
        # Générer des positions avec un layout si non disponibles
        pos = nx.spring_layout(graph, seed=42)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')
    
    # Dessiner les arêtes avec des poids
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    # Dessiner les étiquettes des nœuds avec leurs charges
    node_labels = {}
    for city in graph.nodes:
        load = graph.nodes[city].get('load', 0)
        node_labels[city] = f"{city} ({load})"
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    
    # Dessiner le chemin solution
    if solution_path:
        # Créer des tuples de paires pour les arêtes du chemin
        path_edges = [(solution_path[i], solution_path[i+1]) 
                       for i in range(len(solution_path)-1)]
        # Ajouter l'arête de retour au début
        path_edges.append((solution_path[-1], solution_path[0]))
        
        # Dessiner les arêtes du chemin solution
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, 
                               width=3.0, edge_color='red', arrows=True)
    
    plt.title(f"TSP avec contraintes - Solution pour l'exécution {run_id}")
    plt.axis('off')
    
    # Sauvegarder l'image
    plt.savefig(os.path.join(output_dir, f"tsp_solution_{run_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_graph_to_json(graph_data: Dict[str, Any], output_dir: str, run_id: str) -> str:
    """
    Sauvegarde les données du graphe dans un fichier JSON.
    
    Args:
        graph_data: Données du graphe exportées
        output_dir: Répertoire de sortie
        run_id: Identifiant de l'exécution
        
    Returns:
        Chemin vers le fichier sauvegardé
    """
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, f"graph_{run_id}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graphe sauvegardé dans {filepath}")
    return filepath


def run_single_experiment(args, seed: Optional[int] = None, run_id: str = "001") -> Dict[str, Any]:
    """
    Exécute une expérimentation complète: génération du graphe, 
    résolution TSP, et enregistrement des résultats.
    
    Args:
        args: Arguments de configuration
        seed: Graine aléatoire pour cette exécution
        run_id: Identifiant de l'exécution
        
    Returns:
        Résultat de l'expérimentation
    """
    start_time = time.time()
    print(f"\n--- Début de l'expérimentation (run_id: {run_id}) ---")
    
    if seed is not None:
        print(f"Utilisation de la graine aléatoire: {seed}")
    
    # 1. Génération du graphe
    print("Génération du graphe...")
    generator = TSPGraphGenerator(seed=seed)
    graph = generator.generate_tsp_graph(
        num_cities=args.num_cities,
        geographical=args.geographical,
        min_load=args.min_load,
        max_load=args.max_load,
        dependency_ratio=args.dependency_ratio,
        blocking_ratio=args.blocking_ratio
    )
    
    # Exporter les données du graphe
    graph_data = generator.export_graph_data()
    
    print(f"Graphe généré avec {len(generator.cities)} villes")
    print(f"Nombre d'arêtes: {graph.number_of_edges()}")
    if 'dependencies' in graph.graph:
        print(f"Nombre de dépendances: {len(graph.graph['dependencies'])}")
    if 'blocked_routes' in graph.graph:
        print(f"Nombre de routes bloquées: {len(graph.graph['blocked_routes'])}")
    
    # Sauvegarder le graphe si demandé
    if args.save_graph:
        save_graph_to_json(graph_data, args.output_dir, run_id)
    
    # 2. Résolution avec l'algorithme génétique
    print("\nRésolution du TSP avec l'algorithme génétique...")
    solver = GeneticTSPSolver(
        graph=graph,
        vehicle_capacity=args.vehicle_capacity,
        seed=seed
    )
    
    solution = solver.solve(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        max_iterations=args.max_iterations,
        early_stopping_rounds=args.early_stopping
    )
    
    # Ajouter des informations complémentaires au résultat
    solution['run_id'] = run_id
    solution['graph_info'] = {
        'num_cities': args.num_cities,
        'geographical': args.geographical,
        'dependency_ratio': args.dependency_ratio,
        'blocking_ratio': args.blocking_ratio
    }
    
    # Afficher un résumé de la solution
    print(f"Solution trouvée en {solution['performance_metrics']['iterations']} itérations")
    print(f"Temps d'exécution: {solution['performance_metrics']['execution_time_sec']:.2f} secondes")
    print(f"Coût total: {solution['total_cost']:.2f}")
    print(f"Contraintes respectées: "
          f"Dépendances={solution['constraints']['dependencies_respected']}, "
          f"Capacité={solution['constraints']['capacity_respected']}")
    
    # Visualiser le graphe et la solution si demandé
    if args.visualize:
        print("\nCréation de la visualisation...")
        visualize_graph_and_solution(graph, solution['solution_path'], args.output_dir, run_id)
    
    total_time = time.time() - start_time
    print(f"Temps total de l'expérimentation: {total_time:.2f} secondes")
    print(f"--- Fin de l'expérimentation (run_id: {run_id}) ---\n")
    
    return solution


def run_experiments(args):
    """
    Exécute une série d'expérimentations selon les arguments spécifiés.
    
    Args:
        args: Arguments de configuration
    """
    print("\n==== Démarrage des expérimentations TSP avec contraintes hybrides ====")
    print(f"Configuration: {vars(args)}")
    
    # Initialiser le logger de résultats
    logger = ResultsLogger(output_dir=args.output_dir)
    
    # Déterminer les graines aléatoires pour chaque exécution
    seeds = []
    if args.seed is not None:
        # Si une graine est fournie, l'utiliser pour la première exécution
        # et dériver les suivantes
        base_seed = args.seed
        seeds = [base_seed + i for i in range(args.num_runs)]
    else:
        # Sinon, générer des graines aléatoires
        seeds = [random.randint(1, 10000) for _ in range(args.num_runs)]
    
    # Exécuter les expérimentations
    for i, seed in enumerate(seeds):
        run_id = f"{i+1:03d}"
        
        try:
            result = run_single_experiment(args, seed=seed, run_id=run_id)
            logger.add_result(result)
        except Exception as e:
            print(f"Erreur pendant l'expérimentation {run_id}: {str(e)}")
    
    # Sauvegarder et analyser les résultats
    print("\n==== Sauvegarde et analyse des résultats ====")
    logger.save_to_json(filename="all_results.json")
    logger.save_to_csv(filename="all_results.csv")
    
    # Sauvegarder la meilleure solution
    if logger.results:
        best_result = min(logger.results, key=lambda r: r.get('total_cost', float('inf')))
        logger.save_solution_path(best_result, filename="best_solution.txt")
    
    # Comparer les résultats
    logger.compare_results(metric_key='total_cost')
    
    print("\n==== Expérimentations terminées ====")


if __name__ == "__main__":
    # Parser les arguments
    args = parse_arguments()
    
    # Exécuter les expérimentations
    run_experiments(args)