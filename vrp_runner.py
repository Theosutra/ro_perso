"""
vrp_runner.py

Script principal pour exécuter l'algorithme génétique pour le problème VRP
avec contraintes hybrides.
"""

from vrp_graph_generator import VRPGraphGenerator
from vrp_genetic_solver import GeneticVRPSolver
from results_logger import ResultsLogger

import os
import argparse
import time
import random
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union


def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Runner pour le VRP avec contraintes hybrides')
    
    # Configuration du graphe
    parser.add_argument('--num_cities', type=int, default=20, 
                        help='Nombre de villes dans le graphe (sans compter le dépôt)')
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
    
    # Configuration de la flotte et des véhicules
    parser.add_argument('--num_vehicles', type=int, default=3, 
                        help='Nombre de véhicules dans la flotte')
    parser.add_argument('--min_vehicle_capacity', type=int, default=800, 
                        help='Capacité minimale des véhicules')
    parser.add_argument('--max_vehicle_capacity', type=int, default=1200, 
                        help='Capacité maximale des véhicules')
    parser.add_argument('--heterogeneous_fleet', action='store_true', 
                        help='Utiliser une flotte hétérogène (capacités différentes)')
    parser.add_argument('--depot', type=str, default=None, 
                        help='Nom de la ville servant de dépôt (par défaut: "Depot")')
    
    # Contraintes et fonctionnalités avancées
    parser.add_argument('--time_windows', action='store_true', 
                        help='Ajouter des fenêtres temporelles aux villes')
    parser.add_argument('--use_priorities', action='store_true', 
                        help='Ajouter des niveaux de priorité aux villes')
    
    # Configuration de l'algorithme génétique
    parser.add_argument('--population_size', type=int, default=100, 
                        help='Taille de la population pour l\'algorithme génétique')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                        help='Taux de mutation pour l\'algorithme génétique')
    parser.add_argument('--max_iterations', type=int, default=500, 
                        help='Nombre maximum d\'itérations pour l\'algorithme génétique')
    parser.add_argument('--early_stopping', type=int, default=50, 
                        help='Arrêter si pas d\'amélioration pendant N itérations')
    
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


def visualize_graph_and_solution(graph: nx.Graph, solution_routes: List[List[str]], 
                                output_dir: str, run_id: str):
    """
    Visualise le graphe et les routes de la solution VRP.
    
    Args:
        graph: Graphe NetworkX
        solution_routes: Liste des routes pour chaque véhicule
        output_dir: Répertoire de sortie pour les images
        run_id: Identifiant de l'exécution
    """
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(16, 14))
    
    # Identifier le dépôt
    depot = None
    for node, attrs in graph.nodes(data=True):
        if attrs.get('is_depot', False):
            depot = node
            break
    
    if depot is None and solution_routes and solution_routes[0]:
        depot = solution_routes[0][0]  # Prendre le premier nœud de la première route

    # Vérifier si le graphe a des positions géographiques
    if all('pos' in graph.nodes[city] for city in graph.nodes):
        pos = nx.get_node_attributes(graph, 'pos')
    else:
        # Générer des positions avec un layout si non disponibles
        pos = nx.spring_layout(graph, seed=42)
    
    # Dessiner d'abord toutes les arêtes en gris clair
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.2, edge_color='gray')
    
    # Extraire les charges des villes
    loads = nx.get_node_attributes(graph, 'load')
    
    # Préparer les couleurs des nœuds
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        if node == depot:
            node_colors.append('#FF4500')  # Rouge-orangé pour le dépôt
            node_sizes.append(700)  # Plus grand pour le dépôt
        else:
            node_colors.append('#1E90FF')  # Bleu pour les villes normales
            # Taille proportionnelle à la charge (avec un minimum)
            size = 300 + min(400, loads.get(node, 0) * 2)
            node_sizes.append(size)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors)
    
    # Préparer les étiquettes des nœuds
    node_labels = {}
    for city in graph.nodes():
        load = graph.nodes[city].get('load', 0)
        if city == depot:
            node_labels[city] = f"{city}\n(DÉPÔT)"
        else:
            node_labels[city] = f"{city}\n(Load: {load})"
    
    # Dessiner les étiquettes
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_weight='bold')
    
    # Générer des couleurs distinctes pour chaque véhicule
    num_vehicles = len(solution_routes)
    colormap = cm.get_cmap('tab10', num_vehicles)  # Tab10 a 10 couleurs distinctes
    vehicle_colors = [colormap(i) for i in range(num_vehicles)]
    
    # Dessiner les routes pour chaque véhicule
    legend_elements = []
    
    for i, route in enumerate(solution_routes):
        if len(route) < 2:  # Ignorer les routes trop courtes
            continue
            
        # Créer les arêtes de la route
        route_edges = [(route[j], route[j+1]) for j in range(len(route)-1)]
        
        # Dessiner les arêtes de la route avec une couleur spécifique
        color = vehicle_colors[i]
        nx.draw_networkx_edges(
            graph, pos, edgelist=route_edges,
            width=2.5, alpha=0.8, edge_color=[color],
            arrows=True, arrowsize=15, arrowstyle='-|>'
        )
        
        # Ajouter une entrée à la légende
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color=color, lw=4, label=f'Véhicule {i+1}')
        )
    
    # Si des données de fenêtres temporelles sont disponibles, les afficher
    if 'time_windows' in graph.graph:
        time_windows = graph.graph['time_windows']
        tw_labels = {}
        for city, window in time_windows.items():
            if city != depot:  # Ignorer le dépôt pour la lisibilité
                tw_labels[city] = f"TW: {window[0]}-{window[1]}"
        
        # Positionner les labels de fenêtres temporelles un peu au-dessus des nœuds
        tw_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items() if k in tw_labels}
        nx.draw_networkx_labels(graph, tw_pos, labels=tw_labels, font_size=8, font_color='red')
    
    # Ajouter un titre et une légende
    plt.title(f"Solution VRP - {len(solution_routes)} véhicules - {len(graph.nodes)-1} villes", fontsize=16)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Enlever les axes pour une meilleure visualisation
    plt.axis('off')
    
    # Ajouter un peu d'espace autour du graphe
    plt.tight_layout()
    
    # Sauvegarder l'image
    plt.savefig(os.path.join(output_dir, f"vrp_solution_{run_id}.png"), dpi=300, bbox_inches='tight')
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
    
    filepath = os.path.join(output_dir, f"vrp_graph_{run_id}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graphe sauvegardé dans {filepath}")
    return filepath


def run_single_experiment(args, seed: Optional[int] = None, run_id: str = "001") -> Dict[str, Any]:
    """
    Exécute une expérimentation complète: génération du graphe, 
    résolution VRP, et enregistrement des résultats.
    
    Args:
        args: Arguments de configuration
        seed: Graine aléatoire pour cette exécution
        run_id: Identifiant de l'exécution
        
    Returns:
        Résultat de l'expérimentation
    """
    start_time = time.time()
    print(f"\n--- Début de l'expérimentation VRP (run_id: {run_id}) ---")
    
    if seed is not None:
        print(f"Utilisation de la graine aléatoire: {seed}")
    
    # 1. Génération du graphe spécifique au VRP
    print("Génération du graphe VRP...")
    generator = VRPGraphGenerator(seed=seed)
    graph = generator.generate_vrp_graph(
        num_cities=args.num_cities,
        num_vehicles=args.num_vehicles,
        depot_name=args.depot,
        min_vehicle_capacity=args.min_vehicle_capacity,
        max_vehicle_capacity=args.max_vehicle_capacity,
        homogeneous_fleet=not args.heterogeneous_fleet,
        geographical=args.geographical,
        min_load=args.min_load,
        max_load=args.max_load,
        dependency_ratio=args.dependency_ratio,
        blocking_ratio=args.blocking_ratio,
        time_windows=args.time_windows
    )
    
    # Ajouter des fonctionnalités supplémentaires si demandées
    if args.time_windows:
        generator.add_service_times()
        print("Fenêtres temporelles et temps de service ajoutés")
    
    if args.use_priorities:
        generator.add_priority_levels()
        print("Niveaux de priorité ajoutés")
    
    # Exporter les données du graphe
    graph_data = generator.export_graph_data()
    
    # Récupérer le dépôt et les véhicules
    depot = generator.depot
    vehicles = generator.vehicles
    
    print(f"Graphe généré avec {args.num_cities} villes et 1 dépôt ({depot})")
    print(f"Flotte de {len(vehicles)} véhicules")
    print(f"Nombre d'arêtes: {graph.number_of_edges()}")
    
    if 'dependencies' in graph.graph:
        print(f"Nombre de dépendances: {len(graph.graph['dependencies'])}")
    
    if 'blocked_routes' in graph.graph:
        print(f"Nombre de routes bloquées: {len(graph.graph['blocked_routes'])}")
    
    if args.time_windows:
        print("Fenêtres temporelles activées")
    
    if args.use_priorities:
        print("Niveaux de priorité activés")
    
    # Sauvegarder le graphe si demandé
    if args.save_graph:
        save_graph_to_json(graph_data, args.output_dir, run_id)
    
    # 2. Résolution avec l'algorithme génétique VRP amélioré
    print(f"\nRésolution du VRP avec {args.num_vehicles} véhicules...")
    solver = GeneticVRPSolver(
        graph=graph,
        vehicles=vehicles,
        dependencies=graph.graph.get('dependencies', []),
        depot=depot,
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
        'blocking_ratio': args.blocking_ratio,
        'depot': depot,
        'num_vehicles': args.num_vehicles,
        'has_time_windows': args.time_windows,
        'has_priorities': args.use_priorities,
        'heterogeneous_fleet': args.heterogeneous_fleet
    }
    
    # Afficher un résumé de la solution
    print(f"Solution trouvée en {solution['performance_metrics']['iterations']} itérations")
    print(f"Temps d'exécution: {solution['performance_metrics']['execution_time_sec']:.2f} secondes")
    print(f"Nombre de véhicules utilisés: {solution['fleet_metrics']['total_vehicles_used']} sur {args.num_vehicles}")
    print(f"Coût total: {solution['total_cost']:.2f}")
    print(f"Contraintes respectées: "
          f"Dépendances={solution['constraints']['dependencies_respected']}, "
          f"Capacité={solution['constraints']['capacity_respected']}, "
          f"Toutes villes visitées={solution['constraints']['all_cities_visited']}")
    
    if args.time_windows:
        print(f"Fenêtres temporelles respectées: {solution['constraints'].get('time_windows_respected', False)}")
    
    # Afficher les routes
    print("\nRoutes par véhicule:")
    for veh_result in solution['vehicle_results']:
        print(f"Véhicule {veh_result['vehicle_id']} "
              f"(Capacité: {veh_result['capacity']}): "
              f"{' -> '.join(veh_result['route'])}")
        print(f"  Distance: {veh_result['distance']:.2f}, "
              f"Charge: {veh_result['load']}/{veh_result['capacity']} "
              f"({veh_result['load']/veh_result['capacity']*100:.1f}%)")
        
        if args.time_windows:
            print(f"  Temps d'achèvement: {veh_result.get('completion_time', 0)}")
        
        if args.use_priorities:
            print(f"  Score de priorité: {veh_result.get('priority_score', 0)}")
    
    # Visualiser le graphe et la solution si demandé
    if args.visualize:
        print("\nCréation de la visualisation...")
        visualize_graph_and_solution(graph, solution['solution_routes'], args.output_dir, run_id)
    
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
    print("\n==== Démarrage des expérimentations VRP avec contraintes hybrides ====")
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
            import traceback
            traceback.print_exc()
    
    # Sauvegarder et analyser les résultats
    print("\n==== Sauvegarde et analyse des résultats ====")
    logger.save_to_json(filename="vrp_all_results.json")
    logger.save_to_csv(filename="vrp_all_results.csv")
    
    # Sauvegarder la meilleure solution
    if logger.results:
        best_result = min(logger.results, key=lambda r: r.get('total_cost', float('inf')))
        logger.save_solution_path(best_result, filename="vrp_best_solution.txt")
    
    # Comparer les résultats
    logger.compare_results(metric_key='total_cost')
    
    print("\n==== Expérimentations terminées ====")


if __name__ == "__main__":
    # Parser les arguments
    args = parse_arguments()
    
    # Exécuter les expérimentations
    run_experiments(args)