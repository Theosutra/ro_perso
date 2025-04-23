"""
genetic_solver.py

Ce module implémente un algorithme génétique pour résoudre un problème TSP
avec contraintes hybrides (dépendances, routes bloquées, et contraintes de chargement).
"""

import networkx as nx
import numpy as np
import random
import time
import uuid
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from collections import deque


class GeneticTSPSolver:
    """
    Résolveur de TSP utilisant un algorithme génétique adapté aux contraintes hybrides.
    """
    
    def __init__(self, graph: nx.Graph, dependencies: List[Tuple[str, str]] = None, 
                 vehicle_capacity: int = 1000, seed: Optional[int] = None):
        """
        Initialise le résolveur génétique.
        
        Args:
            graph: Graphe NetworkX représentant le problème TSP
            dependencies: Liste des dépendances entre villes [(ville_A, ville_B), ...]
                          où ville_A doit être visitée avant ville_B
            vehicle_capacity: Capacité maximale du véhicule (pour la contrainte de backpacking)
            seed: Valeur d'initialisation pour la génération aléatoire
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.graph = graph
        self.cities = list(graph.nodes())
        self.num_cities = len(self.cities)
        self.dependencies = dependencies if dependencies else []
        self.vehicle_capacity = vehicle_capacity
        
        # Extraire les dépendances du graphe si elles n'ont pas été fournies explicitement
        if not self.dependencies and 'dependencies' in graph.graph:
            self.dependencies = graph.graph['dependencies']
        
        # Vérifier que le graphe contient des charges de ville
        if not all('load' in graph.nodes[city] for city in self.cities):
            raise ValueError("Le graphe doit avoir des charges ('load') définies pour chaque ville")
        
        # Préparer les structures pour la vérification rapide des dépendances
        self.dependency_dict = self._build_dependency_dict()
        
        # Préparer le dictionnaire de distances entre villes
        self.distances = {(u, v): data['weight'] 
                          for u, v, data in graph.edges(data=True)}
        # Ajouter les distances symétriques
        for (u, v), dist in list(self.distances.items()):
            self.distances[(v, u)] = dist
    
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
    
    def _compute_fitness(self, chromosome: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la valeur de fitness d'un chromosome (chemin).
        La fitness est inversement proportionnelle à la distance totale,
        avec des pénalités pour les violations de contraintes.
        
        Args:
            chromosome: Liste ordonnée de villes représentant un chemin
            
        Returns:
            Tuple (fitness, metrics) où fitness est la valeur de fitness et
            metrics est un dictionnaire de métriques détaillées
        """
        total_distance = 0.0
        current_load = 0
        max_load = 0
        load_distribution = []
        
        # Métriques de contraintes
        all_dependencies_respected = True
        all_capacity_respected = True
        
        # Vérifier les dépendances
        for city in chromosome:
            # Pour chaque ville, vérifier que toutes ses dépendances apparaissent avant elle
            city_dependencies = self.dependency_dict.get(city, [])
            for dep_city in city_dependencies:
                if dep_city not in chromosome[:chromosome.index(city)]:
                    all_dependencies_respected = False
                    break
        
        # Calculer la distance et les charges
        for i in range(len(chromosome)):
            city = chromosome[i]
            
            # Ajouter la charge de la ville
            city_load = self.graph.nodes[city].get('load', 0)
            current_load += city_load
            load_distribution.append(city_load)
            
            # Mettre à jour la charge maximale
            max_load = max(max_load, current_load)
            
            # Vérifier le dépassement de capacité
            if current_load > self.vehicle_capacity:
                all_capacity_respected = False
            
            # Calculer la distance au prochain point
            if i < len(chromosome) - 1:
                next_city = chromosome[i + 1]
                if (city, next_city) in self.distances:
                    total_distance += self.distances[(city, next_city)]
                else:
                    # Grosse pénalité si l'arête est bloquée ou inexistante
                    total_distance += 1000000  # Valeur très élevée pour pénaliser fortement
            
            # Ajouter la distance de retour au point de départ si c'est le dernier point
            if i == len(chromosome) - 1 and (city, chromosome[0]) in self.distances:
                total_distance += self.distances[(city, chromosome[0])]
        
        # Calculer la fitness (inversement proportionnelle à la distance)
        # Avec pénalités pour les violations de contraintes
        fitness = 1.0 / (total_distance + 1.0)  # Éviter la division par zéro
        
        # Appliquer des pénalités pour les violations de contraintes
        if not all_dependencies_respected:
            fitness *= 0.01  # Forte pénalité pour les dépendances non respectées
        
        if not all_capacity_respected:
            fitness *= 0.1  # Pénalité pour dépassement de capacité
        
        # Métriques détaillées
        metrics = {
            "total_distance": total_distance,
            "max_load": max_load,
            "load_distribution": load_distribution,
            "dependencies_respected": all_dependencies_respected,
            "capacity_respected": all_capacity_respected,
            "remaining_capacity": max(0, self.vehicle_capacity - max_load)
        }
        
        return fitness, metrics
    
    def _create_valid_chromosome(self) -> List[str]:
        """
        Crée un chromosome valide respectant les contraintes de dépendance.
        
        Returns:
            Liste ordonnée de villes représentant un chemin valide
        """
        # Construire un graphe de dépendances
        dep_graph = nx.DiGraph()
        dep_graph.add_nodes_from(self.cities)
        
        for city_before, city_after in self.dependencies:
            dep_graph.add_edge(city_before, city_after)
        
        # Utiliser un tri topologique pour obtenir un ordre respectant les dépendances
        try:
            # Générer un ordre topologique aléatoire
            topo_order = list(nx.topological_sort(dep_graph))
            
            # Compléter avec les villes non concernées par des dépendances
            missing_cities = [city for city in self.cities if city not in topo_order]
            random.shuffle(missing_cities)
            valid_order = topo_order + missing_cities
            
            # S'assurer que toutes les villes sont incluses
            if len(valid_order) != self.num_cities:
                # Identifier les villes manquantes
                included = set(valid_order)
                missing = [city for city in self.cities if city not in included]
                # Ajouter les villes manquantes à la fin
                valid_order.extend(missing)
            
            return valid_order
        
        except nx.NetworkXUnfeasible:
            # En cas de cycle dans les dépendances, générer un ordre aléatoire
            # qui pourrait ne pas respecter toutes les dépendances
            chromosome = self.cities.copy()
            random.shuffle(chromosome)
            return chromosome
    
    def _initialize_population(self, population_size: int) -> List[List[str]]:
        """
        Initialise une population de chromosomes valides.
        
        Args:
            population_size: Taille de la population
            
        Returns:
            Liste de chromosomes
        """
        population = []
        
        for _ in range(population_size):
            chromosome = self._create_valid_chromosome()
            population.append(chromosome)
        
        return population
    
    def _tournament_selection(self, population: List[List[str]], fitnesses: List[float],
                             tournament_size: int = 3) -> List[str]:
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
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        
        # Sélectionner le meilleur
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def _order_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Applique l'opérateur de croisement OX (Order Crossover) adapté au TSP.
        
        Args:
            parent1: Premier parent
            parent2: Deuxième parent
            
        Returns:
            Enfant résultant du croisement
        """
        size = len(parent1)
        
        # Choisir deux points de croisement aléatoires
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialiser l'enfant avec les valeurs du premier parent entre les points de croisement
        child = [None] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Remplir le reste avec les valeurs du deuxième parent, en respectant l'ordre
        # et en évitant les doublons
        j = (end + 1) % size  # Position de départ dans l'enfant
        for i in range(size):
            # Position actuelle dans le parent2
            idx = (end + 1 + i) % size
            city = parent2[idx]
            
            # Si la ville n'est pas déjà dans l'enfant, l'ajouter
            if city not in child:
                # Trouver la prochaine position libre dans l'enfant
                while child[j] is not None:
                    j = (j + 1) % size
                
                child[j] = city
                j = (j + 1) % size
        
        # Vérifier que l'enfant contient toutes les villes
        if None in child or len(set(child)) != len(child):
            # En cas de problème, utiliser un des parents
            return parent1 if random.random() < 0.5 else parent2
        
        return child
    
    def _smart_mutation(self, chromosome: List[str], mutation_rate: float) -> List[str]:
        """
        Applique une mutation adaptée au TSP et aux contraintes.
        
        Args:
            chromosome: Chromosome à muter
            mutation_rate: Probabilité de mutation
            
        Returns:
            Chromosome muté
        """
        if random.random() > mutation_rate:
            return chromosome  # Pas de mutation
        
        mutated = chromosome.copy()
        size = len(mutated)
        
        # Tenter plusieurs types de mutations adaptées au TSP
        mutation_type = random.choice(['swap', 'insert', 'reverse'])
        
        if mutation_type == 'swap':
            # Échanger deux villes aléatoires
            i, j = random.sample(range(size), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif mutation_type == 'insert':
            # Insérer une ville à une nouvelle position
            i, j = random.sample(range(size), 2)
            city = mutated.pop(i)
            mutated.insert(j, city)
        
        elif mutation_type == 'reverse':
            # Inverser un segment du chemin
            i, j = sorted(random.sample(range(size), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        
        # Vérifier si la mutation respecte les dépendances
        # Sinon, revenir à l'original
        for city in mutated:
            dependencies = self.dependency_dict.get(city, [])
            city_index = mutated.index(city)
            
            for dep_city in dependencies:
                if dep_city not in mutated[:city_index]:
                    # Dépendance non respectée, annuler la mutation
                    return chromosome
        
        return mutated
    
    def _calculate_population_fitness(self, population: List[List[str]]) -> List[Tuple[float, Dict[str, Any]]]:
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
        Résout le problème TSP avec contraintes en utilisant l'algorithme génétique.
        
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
                child = self._order_crossover(parent1, parent2)
                
                # Mutation
                child = self._smart_mutation(child, mutation_rate)
                
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
        
        # Préparer les résultats
        result = {
            "execution_id": str(uuid.uuid4()),
            "solver": "GeneticAlgorithm",
            "input_parameters": {
                "num_cities": self.num_cities,
                "vehicle_capacity": self.vehicle_capacity,
                "mutation_rate": mutation_rate,
                "population_size": population_size,
                "max_iterations": max_iterations
            },
            "solution_path": best_chromosome,
            "total_cost": best_metrics["total_distance"],
            "performance_metrics": {
                "execution_time_sec": execution_time,
                "iterations": iteration + 1
            },
            "constraints": {
                "dependencies_respected": best_metrics["dependencies_respected"],
                "capacity_respected": best_metrics["capacity_respected"]
            },
            "vehicle_load_stats": {
                "total_volume_delivered": sum(best_metrics["load_distribution"]),
                "maximum_load": best_metrics["max_load"],
                "remaining_capacity": best_metrics["remaining_capacity"],
                "load_distribution": best_metrics["load_distribution"]
            }
        }
        
        return result


# Fonction utilitaire pour tester le solveur
def solve_test_problem(graph, dependencies=None, vehicle_capacity=1000, seed=None):
    """
    Résout un problème TSP de test avec l'algorithme génétique.
    
    Args:
        graph: Graphe NetworkX représentant le problème
        dependencies: Liste des dépendances entre villes
        vehicle_capacity: Capacité maximale du véhicule
        seed: Valeur d'initialisation pour la génération aléatoire
        
    Returns:
        Résultat de la résolution
    """
    solver = GeneticTSPSolver(
        graph=graph,
        dependencies=dependencies,
        vehicle_capacity=vehicle_capacity,
        seed=seed
    )
    
    result = solver.solve(
        population_size=100,
        mutation_rate=0.1,
        max_iterations=500,
        early_stopping_rounds=50
    )
    
    print(f"Solution trouvée en {result['performance_metrics']['iterations']} itérations")
    print(f"Temps d'exécution: {result['performance_metrics']['execution_time_sec']:.2f} secondes")
    print(f"Chemin solution: {result['solution_path']}")
    print(f"Coût total: {result['total_cost']:.2f}")
    print(f"Contraintes respectées: Dépendances={result['constraints']['dependencies_respected']}, "
          f"Capacité={result['constraints']['capacity_respected']}")
    
    return result


if __name__ == "__main__":
    # Test du solveur avec un petit graphe
    import networkx as nx
    
    # Créer un petit graphe pour les tests
    G = nx.Graph()
    cities = ['A', 'B', 'C', 'D', 'E']
    
    # Ajouter les nœuds avec leurs charges
    for city in cities:
        G.add_node(city, load=random.randint(10, 50))
    
    # Ajouter des arêtes avec des poids
    edges = [('A', 'B', 10), ('A', 'C', 15), ('A', 'D', 20), ('A', 'E', 25),
             ('B', 'C', 35), ('B', 'D', 25), ('B', 'E', 30),
             ('C', 'D', 15), ('C', 'E', 20),
             ('D', 'E', 10)]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    # Définir quelques dépendances
    dependencies = [('A', 'C'), ('B', 'E')]
    
    # Résoudre le problème
    result = solve_test_problem(G, dependencies, vehicle_capacity=100, seed=42)