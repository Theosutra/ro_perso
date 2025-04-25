"""
vrp_genetic_solver.py

Ce module implémente un algorithme génétique pour résoudre un problème VRP
(Vehicle Routing Problem) avec contraintes hybrides (dépendances, routes bloquées,
flottes de véhicules et contraintes de chargement).

Le solveur prend en charge les améliorations du générateur VRP, notamment:
- Gestion d'une flotte de véhicules hétérogène
- Fenêtres temporelles
- Temps de service
- Niveaux de priorité
"""

import networkx as nx
import numpy as np
import random
import time
import uuid
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from collections import deque, defaultdict


class GeneticVRPSolver:
    """
    Résolveur de VRP utilisant un algorithme génétique adapté aux contraintes hybrides
    avec une flotte de véhicules.
    """
    
    def __init__(self, graph: nx.Graph, vehicles: List[Dict] = None, 
                 dependencies: List[Tuple[str, str]] = None, 
                 depot: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialise le résolveur génétique pour le VRP.
        
        Args:
            graph: Graphe NetworkX représentant le problème VRP
            vehicles: Liste des véhicules avec leurs caractéristiques
                     [{id: 1, capacity: 1000, depot: "Depot"}, ...]
            dependencies: Liste des dépendances entre villes [(ville_A, ville_B), ...]
                         où ville_A doit être visitée avant ville_B
            depot: Ville de départ/arrivée pour tous les véhicules (si None, identifie automatiquement)
            seed: Valeur d'initialisation pour la génération aléatoire
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Stocker le graphe et extraire les villes
        self.graph = graph
        self.cities = list(graph.nodes())
        self.num_cities = len(self.cities)
        
        # Identifier le dépôt
        if depot is None:
            # 1. Chercher dans les attributs du graphe
            if 'depot' in graph.graph:
                self.depot = graph.graph['depot']
            # 2. Chercher un nœud marqué comme dépôt
            else:
                depot_nodes = [city for city, attrs in graph.nodes(data=True) 
                              if attrs.get('is_depot', False)]
                if depot_nodes:
                    self.depot = depot_nodes[0]
                # 3. Utiliser la première ville par défaut
                else:
                    self.depot = self.cities[0]
        else:
            if depot not in self.cities:
                raise ValueError(f"Le dépôt {depot} n'existe pas dans le graphe")
            self.depot = depot
        
        # Configuration des véhicules
        if vehicles is None:
            # Utiliser la flotte définie dans le graphe
            if 'vehicles' in graph.graph:
                self.vehicles = graph.graph['vehicles']
            # Créer une flotte par défaut
            else:
                # Créer 3 véhicules par défaut
                self.vehicles = [
                    {'id': 1, 'capacity': 1000, 'depot': self.depot},
                    {'id': 2, 'capacity': 1000, 'depot': self.depot},
                    {'id': 3, 'capacity': 1000, 'depot': self.depot}
                ]
        else:
            self.vehicles = vehicles
            
            # S'assurer que tous les véhicules ont un dépôt défini
            for vehicle in self.vehicles:
                if 'depot' not in vehicle or vehicle['depot'] is None:
                    vehicle['depot'] = self.depot
        
        self.num_vehicles = len(self.vehicles)
        
        # Gestion des dépendances
        if dependencies is None:
            # Extraire les dépendances du graphe
            if 'dependencies' in graph.graph:
                self.dependencies = graph.graph['dependencies']
            else:
                self.dependencies = []
        else:
            self.dependencies = dependencies
        
        # Vérifier que le graphe contient des charges (loads) pour les villes
        if not all('load' in graph.nodes[city] for city in self.cities):
            raise ValueError("Le graphe doit avoir des charges ('load') définies pour chaque ville")
        
        # Préparer un dictionnaire pour la vérification rapide des dépendances
        self.dependency_dict = self._build_dependency_dict()
        
        # Extraire les distances du graphe
        self.distances = {(u, v): data['weight'] 
                          for u, v, data in graph.edges(data=True)}
        
        # Ajouter les distances symétriques
        for (u, v), dist in list(self.distances.items()):
            if (v, u) not in self.distances:
                self.distances[(v, u)] = dist
        
        # Fonctionnalités avancées: vérifier leur existence dans le graphe
        
        # Fenêtres temporelles
        self.has_time_windows = 'time_windows' in graph.graph
        if self.has_time_windows:
            self.time_windows = graph.graph['time_windows']
        else:
            self.time_windows = {}
        
        # Temps de service
        self.has_service_times = 'service_times' in graph.graph
        if self.has_service_times:
            self.service_times = graph.graph['service_times']
        else:
            self.service_times = {}
        
        # Niveaux de priorité
        self.has_priorities = 'priorities' in graph.graph
        if self.has_priorities:
            self.priorities = graph.graph['priorities']
        else:
            self.priorities = {}
    
    def _build_dependency_dict(self) -> Dict[str, List[str]]:
        """
        Construit un dictionnaire des dépendances pour une vérification rapide.
        
        Returns:
            Dictionnaire où les clés sont les villes et les valeurs sont des listes
            de villes qui doivent être visitées avant la clé
        """
        dependency_dict = {city: [] for city in self.cities}
        
        for city_before, city_after in self.dependencies:
            dependency_dict[city_after].append(city_before)
        
        return dependency_dict
    
    def _compute_route_metrics(self, route: List[str], vehicle: Dict) -> Dict[str, Any]:
        """
        Calcule les métriques pour une route spécifique d'un véhicule.
        
        Args:
            route: Liste ordonnée de villes représentant la route d'un véhicule
            vehicle: Dictionnaire contenant les caractéristiques du véhicule
            
        Returns:
            Dictionnaire contenant les métriques de la route
        """
        # Si la route est vide, retourner des métriques par défaut
        if not route:
            return {
                "total_distance": 0.0,
                "max_load": 0,
                "load_distribution": [],
                "dependencies_respected": True,
                "capacity_respected": True,
                "time_windows_respected": True,
                "remaining_capacity": vehicle['capacity'],
                "priority_score": 0,
                "completion_time": 0
            }
        
        # Obtenir le dépôt pour ce véhicule
        vehicle_depot = vehicle['depot']
        
        # S'assurer que la route commence et se termine au dépôt
        complete_route = route.copy()
        if complete_route[0] != vehicle_depot:
            complete_route.insert(0, vehicle_depot)
        if complete_route[-1] != vehicle_depot:
            complete_route.append(vehicle_depot)
        
        # Initialiser les métriques
        total_distance = 0.0
        current_load = 0
        max_load = 0
        load_distribution = []
        current_time = 0
        total_priority_score = 0
        
        # Statut des contraintes
        dependencies_respected = True
        capacity_respected = True
        time_windows_respected = True
        
        # Liste des villes visitées dans cette route
        visited_cities = set(complete_route)
        
        # Vérifier les dépendances pour chaque ville de la route
        for idx, city in enumerate(complete_route):
            # Pour chaque ville visitée, vérifier que ses dépendances ont été visitées avant
            for dependency in self.dependency_dict.get(city, []):
                # Si la dépendance est dans cette route, elle doit être visitée avant
                if dependency in visited_cities and dependency not in complete_route[:idx]:
                    dependencies_respected = False
                    break
        
        # Calculer les métriques le long de la route
        for i in range(len(complete_route) - 1):
            current_city = complete_route[i]
            next_city = complete_route[i + 1]
            
            # Ajouter la charge de la ville (sauf pour le dépôt)
            if current_city != vehicle_depot:
                city_load = self.graph.nodes[current_city].get('load', 0)
                current_load += city_load
                load_distribution.append(city_load)
                
                # Ajouter le score de priorité si applicable
                if self.has_priorities and current_city in self.priorities:
                    # Les valeurs faibles indiquent une haute priorité, inverser pour le score
                    priority = self.priorities[current_city]
                    max_priority = max(self.priorities.values()) if self.priorities else 1
                    priority_score = max_priority - priority + 1
                    total_priority_score += priority_score
            
            # Mettre à jour la charge maximale
            max_load = max(max_load, current_load)
            
            # Vérifier le respect de la capacité
            if current_load > vehicle['capacity']:
                capacity_respected = False
            
            # Calculer la distance et le temps de trajet
            if (current_city, next_city) in self.distances:
                distance = self.distances[(current_city, next_city)]
                total_distance += distance
                
                # Mettre à jour le temps courant (distance = temps pour simplifier)
                travel_time = distance
                current_time += travel_time
            else:
                # Pénalité élevée si l'arête est bloquée ou n'existe pas
                penalty = 1000000
                total_distance += penalty
                current_time += penalty
            
            # Ajouter le temps de service si applicable
            if self.has_service_times and current_city in self.service_times:
                service_time = self.service_times[current_city]
                current_time += service_time
            
            # Vérifier les fenêtres temporelles pour la ville suivante
            if self.has_time_windows and next_city in self.time_windows:
                earliest, latest = self.time_windows[next_city]
                
                # Si on arrive avant la fenêtre d'ouverture, attendre
                if current_time < earliest:
                    current_time = earliest
                
                # Si on arrive après la fermeture, contrainte non respectée
                if current_time > latest:
                    time_windows_respected = False
        
        # Résultats de l'évaluation de la route
        return {
            "total_distance": total_distance,
            "max_load": max_load,
            "load_distribution": load_distribution,
            "dependencies_respected": dependencies_respected,
            "capacity_respected": capacity_respected,
            "time_windows_respected": time_windows_respected,
            "remaining_capacity": max(0, vehicle['capacity'] - max_load),
            "priority_score": total_priority_score,
            "completion_time": current_time
        }
    
    def _compute_fitness(self, chromosome: List[List[str]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la valeur de fitness d'un chromosome (ensemble de routes pour tous les véhicules).
        
        Args:
            chromosome: Liste de routes, chaque route étant une liste ordonnée de villes
            
        Returns:
            Tuple (fitness, metrics) où fitness est la valeur de fitness et
            metrics est un dictionnaire de métriques détaillées
        """
        # Initialiser les métriques globales
        total_distance = 0.0
        all_dependencies_respected = True
        all_capacity_respected = True
        all_time_windows_respected = True
        total_priority_score = 0
        
        # Métriques par véhicule
        vehicle_metrics = []
        
        # Identifier toutes les villes visitées
        visited_cities = set()
        for route in chromosome:
            for city in route:
                if city != self.depot:
                    visited_cities.add(city)
        
        # Villes qui doivent être visitées (toutes sauf le dépôt)
        required_cities = set(city for city in self.cities 
                          if city != self.depot and not self.graph.nodes[city].get('is_depot', False))
        
        # Vérifier que toutes les villes sont bien visitées
        all_cities_visited = visited_cities.issuperset(required_cities)
        
        # Vérifier qu'aucune ville n'est visitée plus d'une fois
        city_visit_count = defaultdict(int)
        for route in chromosome:
            for city in route:
                if city != self.depot:
                    city_visit_count[city] += 1
        
        no_duplicate_visits = all(city_visit_count[city] == 1 
                                for city in required_cities 
                                if city in city_visit_count)
        
        # Calculer les métriques pour chaque route/véhicule
        for i, route in enumerate(chromosome):
            # S'assurer qu'on a assez de véhicules
            if i >= len(self.vehicles):
                break
                
            vehicle = self.vehicles[i]
            route_metrics = self._compute_route_metrics(route, vehicle)
            vehicle_metrics.append(route_metrics)
            
            # Agréger les métriques globales
            total_distance += route_metrics["total_distance"]
            all_dependencies_respected = all_dependencies_respected and route_metrics["dependencies_respected"]
            all_capacity_respected = all_capacity_respected and route_metrics["capacity_respected"]
            all_time_windows_respected = all_time_windows_respected and route_metrics["time_windows_respected"]
            total_priority_score += route_metrics["priority_score"]
        
        # Calculer la fitness de base (inversement proportionnelle à la distance)
        fitness = 1.0 / (total_distance + 1.0)  # Éviter division par zéro
        
        # Appliquer les pénalités pour les contraintes non respectées
        if not all_dependencies_respected:
            fitness *= 0.01  # Forte pénalité
        
        if not all_capacity_respected:
            fitness *= 0.1  # Pénalité modérée
        
        if self.has_time_windows and not all_time_windows_respected:
            fitness *= 0.1  # Pénalité modérée
        
        if not all_cities_visited or not no_duplicate_visits:
            fitness *= 0.01  # Forte pénalité
        
        # Bonus pour priorités bien gérées
        if self.has_priorities and total_priority_score > 0:
            max_possible_score = len(required_cities) * (max(self.priorities.values()) if self.priorities else 1)
            priority_ratio = total_priority_score / max(1, max_possible_score)
            fitness *= (1 + 0.1 * priority_ratio)  # Petit bonus
        
        # Métriques complètes
        metrics = {
            "total_distance": total_distance,
            "dependencies_respected": all_dependencies_respected,
            "capacity_respected": all_capacity_respected,
            "time_windows_respected": all_time_windows_respected,
            "all_cities_visited": all_cities_visited,
            "no_duplicate_visits": no_duplicate_visits,
            "priority_score": total_priority_score,
            "vehicle_metrics": vehicle_metrics
        }
        
        return fitness, metrics
    
    def _create_initial_route(self, available_cities: List[str], 
                            respect_dependencies: bool = True) -> List[str]:
        """
        Crée une route initiale valide pour un véhicule.
        
        Args:
            available_cities: Liste des villes disponibles
            respect_dependencies: Si True, essaie de respecter les contraintes de dépendance
            
        Returns:
            Liste ordonnée de villes représentant une route valide
        """
        if not available_cities:
            return []
        
        route = []
        remaining_cities = available_cities.copy()
        
        if respect_dependencies:
            # Créer un graphe de dépendances pour ces villes
            temp_dep_graph = nx.DiGraph()
            temp_dep_graph.add_nodes_from(remaining_cities)
            
            # Ajouter les arêtes de dépendance
            for city_before, city_after in self.dependencies:
                if city_before in remaining_cities and city_after in remaining_cities:
                    temp_dep_graph.add_edge(city_before, city_after)
            
            try:
                # Faire un tri topologique pour respecter les dépendances
                route = list(nx.topological_sort(temp_dep_graph))
                
                # Ajouter les villes sans dépendances
                missing = [city for city in remaining_cities if city not in route]
                random.shuffle(missing)
                route.extend(missing)
            except nx.NetworkXUnfeasible:
                # En cas de cycle dans les dépendances, utiliser un ordre aléatoire
                route = remaining_cities.copy()
                random.shuffle(route)
        else:
            # Ordre aléatoire simple
            route = remaining_cities.copy()
            random.shuffle(route)
        
        return route
    
    def _create_valid_chromosome(self) -> List[List[str]]:
        """
        Crée un chromosome valide (ensemble de routes pour tous les véhicules).
        
        Returns:
            Liste de routes, chaque route étant une liste ordonnée de villes
        """
        # Villes à visiter (toutes sauf le dépôt)
        cities_to_visit = [city for city in self.cities 
                         if city != self.depot and not self.graph.nodes[city].get('is_depot', False)]
        
        random.shuffle(cities_to_visit)
        
        # Routes pour chaque véhicule
        vehicle_routes = [[] for _ in range(self.num_vehicles)]
        
        # Stratégie simple: répartition équitable des villes entre véhicules
        cities_per_vehicle = len(cities_to_visit) // self.num_vehicles
        remaining = len(cities_to_visit) % self.num_vehicles
        
        start_idx = 0
        for v_idx in range(self.num_vehicles):
            # Nombre de villes pour ce véhicule
            n_cities = cities_per_vehicle + (1 if v_idx < remaining else 0)
            
            if n_cities > 0:
                # Villes affectées à ce véhicule
                assigned_cities = cities_to_visit[start_idx:start_idx + n_cities]
                start_idx += n_cities
                
                # Créer une route optimisée pour ces villes
                route = self._create_initial_route(assigned_cities, respect_dependencies=True)
                vehicle_routes[v_idx] = route
        
        # Optimisation supplémentaire des routes en fonction des dépendances globales
        self._optimize_routes_for_dependencies(vehicle_routes)
        
        # Si nous avons des fenêtres temporelles, optimiser en conséquence
        if self.has_time_windows:
            self._optimize_routes_for_time_windows(vehicle_routes)
        
        return vehicle_routes
    
    def _optimize_routes_for_dependencies(self, vehicle_routes: List[List[str]]) -> None:
        """
        Optimise la répartition des villes entre les routes pour mieux respecter les dépendances.
        Modifie les routes en place.
        
        Args:
            vehicle_routes: Liste de routes à optimiser
        """
        max_attempts = 20
        
        for _ in range(max_attempts):
            dependency_violations = []
            
            # Identifier les violations de dépendances entre routes
            for v_idx, route in enumerate(vehicle_routes):
                route_cities = set(route)
                
                for city in route:
                    # Vérifier si des dépendances de cette ville sont dans d'autres routes
                    for dep_city in self.dependency_dict.get(city, []):
                        if dep_city not in route_cities:
                            # Chercher dans quelle route se trouve la dépendance
                            for other_idx, other_route in enumerate(vehicle_routes):
                                if dep_city in other_route and other_idx != v_idx:
                                    # Violation: dépendance dans une autre route
                                    dependency_violations.append((dep_city, city, other_idx, v_idx))
            
            if not dependency_violations:
                break  # Pas de violations, on a terminé
            
            # Corriger une violation aléatoire
            if dependency_violations:
                dep_city, city, from_idx, to_idx = random.choice(dependency_violations)
                
                # Déplacer la ville dépendante vers la route de la dépendance
                vehicle_routes[from_idx].remove(dep_city)
                
                # Insérer avant la ville dépendante
                if city in vehicle_routes[to_idx]:
                    city_pos = vehicle_routes[to_idx].index(city)
                    vehicle_routes[to_idx].insert(city_pos, dep_city)
                else:
                    # Si la ville n'est plus dans cette route
                    vehicle_routes[to_idx].append(dep_city)
    
    def _optimize_routes_for_time_windows(self, vehicle_routes: List[List[str]]) -> None:
        """
        Optimise les routes en fonction des fenêtres temporelles.
        Modifie les routes en place.
        
        Args:
            vehicle_routes: Liste de routes à optimiser
        """
        # Trier les villes de chaque route selon leurs fenêtres temporelles
        for idx, route in enumerate(vehicle_routes):
            if not route:
                continue
                
            # Créer un dictionnaire ville -> fenêtre temporelle (ou valeur par défaut)
            time_window_dict = {}
            for city in route:
                if city in self.time_windows:
                    time_window_dict[city] = self.time_windows[city][0]  # Utiliser le début de la fenêtre
                else:
                    time_window_dict[city] = float('inf')  # Valeur élevée pour les villes sans contrainte
            
            # Trier les villes par heure de début de fenêtre temporelle
            sorted_route = sorted(route, key=lambda city: time_window_dict[city])
            
            # Remplacer la route par la version triée
            vehicle_routes[idx] = sorted_route
    
    def _initialize_population(self, population_size: int) -> List[List[List[str]]]:
        """
        Initialise une population de chromosomes valides.
        
        Args:
            population_size: Taille de la population
            
        Returns:
            Liste de chromosomes (chaque chromosome est une liste de routes)
        """
        population = []
        
        for _ in range(population_size):
            chromosome = self._create_valid_chromosome()
            population.append(chromosome)
        
        return population
    
    def _tournament_selection(self, population: List[List[List[str]]], fitnesses: List[float],
                             tournament_size: int = 3) -> List[List[str]]:
        """
        Sélectionne un chromosome par tournoi.
        
        Args:
            population: Liste de chromosomes
            fitnesses: Valeurs de fitness correspondantes
            tournament_size: Nombre de participants au tournoi
            
        Returns:
            Chromosome sélectionné
        """
        # Sélectionner des individus aléatoires pour le tournoi
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        
        # Choisir le meilleur individu du tournoi
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def _crossover(self, parent1: List[List[str]], parent2: List[List[str]]) -> List[List[str]]:
        """
        Applique un opérateur de croisement adapté au VRP.
        
        Args:
            parent1: Premier parent (liste de routes)
            parent2: Deuxième parent (liste de routes)
            
        Returns:
            Enfant résultant du croisement
        """
        # Liste de toutes les villes à visiter (hors dépôt)
        all_cities = set()
        for route in parent1 + parent2:
            for city in route:
                if city != self.depot:
                    all_cities.add(city)
        
        # Initialiser les routes vides pour l'enfant
        child_routes = [[] for _ in range(self.num_vehicles)]
        
        # Ensemble des villes déjà attribuées
        assigned_cities = set()
        
        # Sélectionner de manière aléatoire des routes complètes
        # alternativement depuis les deux parents
        for vehicle_idx in range(self.num_vehicles):
            # Choisir entre les deux parents
            parent = parent1 if random.random() < 0.5 else parent2
            
            # Vérifier si l'index existe dans ce parent
            if vehicle_idx < len(parent):
                selected_route = parent[vehicle_idx].copy()
            else:
                selected_route = []
            
            # Filtrer les villes déjà attribuées
            valid_route = [city for city in selected_route if city not in assigned_cities]
            child_routes[vehicle_idx] = valid_route
            
            # Mettre à jour l'ensemble des villes attribuées
            assigned_cities.update(valid_route)
        
        # Distribuer les villes non attribuées
        unassigned_cities = all_cities - assigned_cities
        if unassigned_cities:
            city_list = list(unassigned_cities)
            random.shuffle(city_list)
            
            # Répartir les villes entre les véhicules
            for city in city_list:
                # Choisir un véhicule de manière aléatoire
                vehicle_idx = random.randrange(self.num_vehicles)
                child_routes[vehicle_idx].append(city)
        
        # Optimiser les routes pour les dépendances
        self._optimize_routes_for_dependencies(child_routes)
        
        # Optimiser pour les fenêtres temporelles si nécessaire
        if self.has_time_windows:
            self._optimize_routes_for_time_windows(child_routes)
        
        return child_routes
    
    def _mutation(self, chromosome: List[List[str]], mutation_rate: float) -> List[List[str]]:
        """
        Applique une mutation adaptée au VRP.
        
        Args:
            chromosome: Chromosome à muter (liste de routes)
            mutation_rate: Probabilité de mutation
            
        Returns:
            Chromosome muté
        """
        if random.random() > mutation_rate:
            return chromosome  # Pas de mutation
        
        # Copier le chromosome pour la mutation
        mutated = [route.copy() for route in chromosome]
        
        # Sélectionner aléatoirement un type de mutation
        mutation_type = random.choice(['swap', 'transfer', 'invert', 'relocate'])
        
        if mutation_type == 'swap':
            # Échanger deux villes entre deux routes différentes
            non_empty_routes = [i for i, route in enumerate(mutated) if route]
            
            if len(non_empty_routes) >= 2:
                # Sélectionner deux routes non vides
                route1_idx, route2_idx = random.sample(non_empty_routes, 2)
                
                if mutated[route1_idx] and mutated[route2_idx]:
                    # Sélectionner une ville aléatoire dans chaque route
                    city1_idx = random.randrange(len(mutated[route1_idx]))
                    city2_idx = random.randrange(len(mutated[route2_idx]))
                    
                    # Échanger les villes
                    mutated[route1_idx][city1_idx], mutated[route2_idx][city2_idx] = \
                        mutated[route2_idx][city2_idx], mutated[route1_idx][city1_idx]
        
        elif mutation_type == 'transfer':
            # Transférer une ville d'une route à une autre
            non_empty_routes = [i for i, route in enumerate(mutated) if route]
            
            if non_empty_routes:
                # Sélectionner une route source non vide
                from_route_idx = random.choice(non_empty_routes)
                
                # Sélectionner une route destination (peut être vide)
                to_route_idx = random.randrange(self.num_vehicles)
                
                if from_route_idx != to_route_idx and mutated[from_route_idx]:
                    # Sélectionner une ville à transférer
                    city_idx = random.randrange(len(mutated[from_route_idx]))
                    city = mutated[from_route_idx].pop(city_idx)
                    
                    # Insérer la ville dans la route destination
                    insert_pos = random.randint(0, len(mutated[to_route_idx]))
                    mutated[to_route_idx].insert(insert_pos, city)
        
        elif mutation_type == 'invert':
            # Inverser un segment dans une route
            non_empty_routes = [i for i, route in enumerate(mutated) if len(route) >= 2]
            
            if non_empty_routes:
                # Sélectionner une route non vide
                route_idx = random.choice(non_empty_routes)
                route = mutated[route_idx]
                
                # Sélectionner deux positions pour définir le segment
                pos1, pos2 = sorted(random.sample(range(len(route)), 2))
                
                # Inverser le segment
                mutated[route_idx][pos1:pos2+1] = reversed(route[pos1:pos2+1])
        
        elif mutation_type == 'relocate':
            # Déplacer une ville à une nouvelle position dans la même route
            non_empty_routes = [i for i, route in enumerate(mutated) if len(route) >= 2]
            
            if non_empty_routes:
                # Sélectionner une route non vide
                route_idx = random.choice(non_empty_routes)
                route = mutated[route_idx]
                
                # Sélectionner une ville et sa nouvelle position
                old_pos = random.randrange(len(route))
                city = route.pop(old_pos)
                
                # Insérer à une nouvelle position
                new_pos = random.randrange(len(route) + 1)
                route.insert(new_pos, city)
        
        # Vérifier si la mutation respecte les dépendances
        _, metrics = self._compute_fitness(mutated)
        
        if not metrics["dependencies_respected"]:
            # Si les dépendances ne sont pas respectées, tenter de les optimiser
            self._optimize_routes_for_dependencies(mutated)
            
            # Vérifier à nouveau
            _, new_metrics = self._compute_fitness(mutated)
            
            if not new_metrics["dependencies_respected"]:
                # Si toujours pas respectées, revenir à l'original
                return chromosome
        
        # Optimiser pour les fenêtres temporelles si nécessaire
        if self.has_time_windows:
            self._optimize_routes_for_time_windows(mutated)
        
        return mutated
    
    def _calculate_population_fitness(self, population: List[List[List[str]]]) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Calcule la fitness et les métriques pour toute la population.
        
        Args:
            population: Liste de chromosomes
            
        Returns:
            Liste de tuples (fitness, metrics) pour chaque chromosome
        """
        results = []
        for chromosome in population:
            fitness, metrics = self._compute_fitness(chromosome)
            results.append((fitness, metrics))
        
        return results
    
    def _check_termination(self, best_fitness_history: List[float], 
                          iteration: int, max_iterations: int,
                          early_stopping_rounds: int) -> bool:
        """
        Vérifie si l'algorithme doit s'arrêter.
        
        Args:
            best_fitness_history: Historique des meilleures fitness
            iteration: Itération actuelle
            max_iterations: Nombre maximum d'itérations
            early_stopping_rounds: Nombre d'itérations sans amélioration avant arrêt
            
        Returns:
            True si l'algorithme doit s'arrêter, False sinon
        """
        # Vérifier si nous avons atteint le nombre maximal d'itérations
        if iteration >= max_iterations:
            return True
        
        # Vérifier l'arrêt anticipé si nous avons assez d'historique
        if len(best_fitness_history) > early_stopping_rounds:
            # Vérifier si la fitness s'est améliorée récemment
            recent_best = best_fitness_history[-early_stopping_rounds:]
            if all(abs(recent_best[i] - recent_best[0]) < 1e-6 for i in range(1, len(recent_best))):
                return True  # Pas d'amélioration significative
        
        return False
    
    def solve(self, population_size: int = 100, mutation_rate: float = 0.1,
             max_iterations: int = 1000, early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Résout le problème VRP avec contraintes en utilisant l'algorithme génétique.
        
        Args:
            population_size: Taille de la population
            mutation_rate: Taux de mutation
            max_iterations: Nombre maximum d'itérations
            early_stopping_rounds: Nombre d'itérations sans amélioration avant arrêt anticipé
            
        Returns:
            Dictionnaire contenant la solution et les métriques
        """
        start_time = time.time()
        
        # Initialiser la population
        population = self._initialize_population(population_size)
        
        # Calculer la fitness initiale
        fitness_metrics = self._calculate_population_fitness(population)
        fitnesses = [fm[0] for fm in fitness_metrics]
        
        # Suivre le meilleur individu
        best_index = np.argmax(fitnesses)
        best_chromosome = population[best_index]
        best_fitness = fitnesses[best_index]
        best_metrics = fitness_metrics[best_index][1]
        
        # Historique pour l'arrêt anticipé
        best_fitness_history = [best_fitness]
        
        # Boucle principale de l'algorithme génétique
        for iteration in range(max_iterations):
            # Créer une nouvelle génération
            new_population = []
            
            # Élitisme: garder le meilleur individu
            new_population.append(best_chromosome)
            
            # Générer de nouveaux individus
            for _ in range(population_size - 1):
                # Sélection des parents
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # Croisement
                child = self._crossover(parent1, parent2)
                
                # Mutation
                child = self._mutation(child, mutation_rate)
                
                new_population.append(child)
            
            # Mettre à jour la population
            population = new_population
            
            # Calculer la nouvelle fitness
            fitness_metrics = self._calculate_population_fitness(population)
            fitnesses = [fm[0] for fm in fitness_metrics]
            
            # Mettre à jour le meilleur individu
            current_best_index = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_index]
            
            if current_best_fitness > best_fitness:
                best_index = current_best_index
                best_chromosome = population[best_index]
                best_fitness = current_best_fitness
                best_metrics = fitness_metrics[best_index][1]
            
            best_fitness_history.append(best_fitness)
            
            # Vérifier l'arrêt anticipé
            if self._check_termination(best_fitness_history, iteration, max_iterations, early_stopping_rounds):
                break
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        
        # Préparer les routes complètes avec le dépôt pour chaque véhicule
        complete_routes = []
        for route in best_chromosome:
            if not route:  # Ignorer les routes vides
                continue
                
            # Ajouter le dépôt au début et à la fin de chaque route
            complete_route = [self.depot] + route + [self.depot] \
                if route[0] != self.depot and route[-1] != self.depot \
                else route
            
            complete_routes.append(complete_route)
        
        # Calculer les métriques par véhicule
        vehicle_results = []
        for i, route in enumerate(best_chromosome):
            if i >= len(self.vehicles) or not route:  # Ignorer les routes vides
                continue
                
            vehicle = self.vehicles[i]
            route_metrics = self._compute_route_metrics(route, vehicle)
            vehicle_results.append({
                "vehicle_id": vehicle['id'],
                "route": [self.depot] + route + [self.depot] if route[0] != self.depot and route[-1] != self.depot else route,
                "distance": route_metrics["total_distance"],
                "load": route_metrics["max_load"],
                "capacity": vehicle['capacity'],
                "capacity_respected": route_metrics["capacity_respected"],
                "dependencies_respected": route_metrics["dependencies_respected"],
                "time_windows_respected": route_metrics.get("time_windows_respected", True),
                "priority_score": route_metrics.get("priority_score", 0),
                "completion_time": route_metrics.get("completion_time", 0)
            })
        
        # Préparer les résultats
        result = {
            "execution_id": str(uuid.uuid4()),
            "solver": "GeneticVRPAlgorithm",
            "input_parameters": {
                "num_cities": self.num_cities - 1,  # Ne pas compter le dépôt
                "num_vehicles": self.num_vehicles,
                "vehicle_capacities": [v['capacity'] for v in self.vehicles],
                "depot": self.depot,
                "mutation_rate": mutation_rate,
                "population_size": population_size,
                "max_iterations": max_iterations,
                "has_time_windows": self.has_time_windows,
                "has_priorities": self.has_priorities
            },
            "solution_routes": complete_routes,
            "total_cost": best_metrics["total_distance"],
            "performance_metrics": {
                "execution_time_sec": execution_time,
                "iterations": iteration + 1
            },
            "constraints": {
                "dependencies_respected": best_metrics["dependencies_respected"],
                "capacity_respected": best_metrics["capacity_respected"],
                "all_cities_visited": best_metrics["all_cities_visited"],
                "no_duplicate_visits": best_metrics["no_duplicate_visits"],
                "time_windows_respected": best_metrics.get("time_windows_respected", True)
            },
            "vehicle_results": vehicle_results,
            "fleet_metrics": {
                "total_vehicles_used": len([r for r in best_chromosome if r]),
                "average_route_length": sum(len(r) for r in best_chromosome) / max(1, len([r for r in best_chromosome if r])),
                "max_route_length": max([len(r) for r in best_chromosome if r], default=0),
                "min_route_length": min([len(r) for r in best_chromosome if r], default=0),
                "priority_score": best_metrics.get("priority_score", 0)
            }
        }
        
        return result


# Fonction utilitaire pour tester le solveur
def solve_test_problem(graph, vehicles=None, dependencies=None, depot=None, seed=None):
    """
    Résout un problème VRP de test avec l'algorithme génétique.
    
    Args:
        graph: Graphe NetworkX représentant le problème
        vehicles: Liste des véhicules et leurs caractéristiques
        dependencies: Liste des dépendances entre villes
        depot: Ville de départ/arrivée (dépôt)
        seed: Valeur d'initialisation pour la génération aléatoire
        
    Returns:
        Résultat de la résolution
    """
    solver = GeneticVRPSolver(
        graph=graph,
        vehicles=vehicles,
        dependencies=dependencies,
        depot=depot,
        seed=seed
    )
    
    result = solver.solve(
        population_size=100,
        mutation_rate=0.1,
        max_iterations=500,
        early_stopping_rounds=50
    )
    
    print(f"\nSolution trouvée en {result['performance_metrics']['iterations']} itérations")
    print(f"Temps d'exécution: {result['performance_metrics']['execution_time_sec']:.2f} secondes")
    print(f"Nombre de véhicules utilisés: {result['fleet_metrics']['total_vehicles_used']} sur {solver.num_vehicles}")
    print(f"Coût total: {result['total_cost']:.2f}")
    print(f"Contraintes respectées:")
    print(f"  - Dépendances: {result['constraints']['dependencies_respected']}")
    print(f"  - Capacité: {result['constraints']['capacity_respected']}")
    print(f"  - Toutes villes visitées: {result['constraints']['all_cities_visited']}")
    
    if solver.has_time_windows:
        print(f"  - Fenêtres temporelles: {result['constraints'].get('time_windows_respected', False)}")
    
    if solver.has_priorities:
        print(f"Score de priorité: {result['fleet_metrics'].get('priority_score', 0)}")
    
    # Afficher chaque route
    print("\nRoutes par véhicule:")
    for veh_result in result['vehicle_results']:
        print(f"Véhicule {veh_result['vehicle_id']}: "
              f"{' -> '.join(veh_result['route'])}")
        print(f"  Distance: {veh_result['distance']:.2f}, "
              f"Charge: {veh_result['load']}/{veh_result['capacity']}")
        
        if solver.has_time_windows:
            print(f"  Temps d'achèvement: {veh_result.get('completion_time', 0)}")
    
    return result


if __name__ == "__main__":
    # Test du solveur avec un petit graphe généré par le VRPGraphGenerator
    from vrp_graph_generator import VRPGraphGenerator
    
    print("Génération d'un graphe VRP de test...")
    generator = VRPGraphGenerator(seed=42)
    graph = generator.generate_vrp_graph(
        num_cities=10,
        num_vehicles=3,
        geographical=True,
        time_windows=True,
        homogeneous_fleet=False
    )
    
    # Ajouter des temps de service
    generator.add_service_times()
    
    # Ajouter des priorités
    generator.add_priority_levels()
    
    print("Résolution du problème VRP avec contraintes hybrides...")
    result = solve_test_problem(
        graph,
        vehicles=generator.vehicles,
        dependencies=generator.graph.graph.get('dependencies', []),
        depot=generator.depot,
        seed=42
    )