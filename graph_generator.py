"""
graph_generator.py

Ce module permet de générer des graphes aléatoires pour tester l'algorithme TSP
avec contraintes hybrides (dépendances, routes bloquées, et contraintes de chargement).
"""

import networkx as nx
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Set, Optional, Union


class TSPGraphGenerator:
    """
    Générateur de graphes pour le problème TSP avec contraintes hybrides.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialise le générateur de graphes.
        
        Args:
            seed: Valeur d'initialisation pour la génération aléatoire (reproductibilité)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.cities = []
        self.graph = None
    
    def generate_complete_graph(self, num_cities: int, min_distance: float = 1.0, 
                               max_distance: float = 100.0, geographical: bool = False) -> nx.Graph:
        """
        Génère un graphe complet avec des distances aléatoires ou géographiques.
        
        Args:
            num_cities: Nombre de villes (nœuds)
            min_distance: Distance minimale entre deux villes
            max_distance: Distance maximale entre deux villes
            geographical: Si True, génère des coordonnées et calcule des distances euclidiennes
                          Si False, génère des distances aléatoires directement
        
        Returns:
            Un graphe NetworkX complet avec des distances comme attributs d'arêtes
        """
        # Créer un nouveau graphe
        G = nx.Graph()
        
        # Générer les noms des villes (A, B, C, ... ou ville_1, ville_2, ...)
        if num_cities <= 26:
            self.cities = [chr(65 + i) for i in range(num_cities)]  # A, B, C, ...
        else:
            self.cities = [f"ville_{i+1}" for i in range(num_cities)]

        # Ajouter les nœuds au graphe
        G.add_nodes_from(self.cities)
        
        if geographical:
            # Générer des coordonnées aléatoires pour chaque ville
            positions = {}
            for city in self.cities:
                positions[city] = (random.uniform(0, 1000), random.uniform(0, 1000))
            
            # Calculer les distances euclidiennes entre toutes les paires de villes
            for i, city1 in enumerate(self.cities):
                for city2 in self.cities[i+1:]:
                    x1, y1 = positions[city1]
                    x2, y2 = positions[city2]
                    # Distance euclidienne
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    G.add_edge(city1, city2, weight=distance)
            
            # Enregistrer les positions comme attributs des nœuds
            nx.set_node_attributes(G, positions, 'pos')
            
        else:
            # Générer des distances aléatoires entre toutes les paires de villes
            for i, city1 in enumerate(self.cities):
                for city2 in self.cities[i+1:]:
                    distance = random.uniform(min_distance, max_distance)
                    G.add_edge(city1, city2, weight=distance)
        
        self.graph = G
        return G
    
    def add_city_loads(self, min_load: int = 10, max_load: int = 100) -> Dict[str, int]:
        """
        Ajoute des charges aléatoires à chaque ville (contrainte de backpacking).
        
        Args:
            min_load: Charge minimale pour une ville
            max_load: Charge maximale pour une ville
            
        Returns:
            Dictionnaire des charges associées à chaque ville
        """
        if not self.graph:
            raise ValueError("Il faut d'abord générer un graphe avec generate_complete_graph()")
        
        loads = {}
        for city in self.cities:
            loads[city] = random.randint(min_load, max_load)
        
        # Ajouter les charges comme attributs des nœuds
        nx.set_node_attributes(self.graph, loads, 'load')
        
        return loads
    
    def add_dependencies(self, num_dependencies: Optional[int] = None, 
                        dependency_ratio: float = 0.2) -> List[Tuple[str, str]]:
        """
        Ajoute des contraintes de dépendance entre les villes (A doit être visité avant B).
        
        Args:
            num_dependencies: Nombre de dépendances à générer, 
                              si None, utilise dependency_ratio
            dependency_ratio: Ratio du nombre de dépendances par rapport au nombre total de villes
        
        Returns:
            Liste des dépendances sous forme de tuples (ville_avant, ville_après)
        """
        if not self.graph:
            raise ValueError("Il faut d'abord générer un graphe avec generate_complete_graph()")
        
        # Déterminer le nombre de dépendances à créer
        if num_dependencies is None:
            num_cities = len(self.cities)
            max_possible_dependencies = num_cities * (num_cities - 1) // 2  # Nombre max de paires possibles
            num_dependencies = min(int(dependency_ratio * num_cities), max_possible_dependencies)
        
        # Créer un graphe dirigé acyclique (DAG) pour représenter les dépendances
        # et éviter les cycles qui rendraient le problème insoluble
        dag = nx.DiGraph()
        dag.add_nodes_from(self.cities)
        
        dependencies = []
        cities_copy = self.cities.copy()
        
        # Mélanger la liste pour générer des dépendances aléatoires
        random.shuffle(cities_copy)
        
        # Générer un ordre topologique aléatoire
        for i in range(num_dependencies):
            # Prenons deux villes aléatoires mais veillons à respecter un ordre possible
            # (éviter les cycles)
            valid_pairs = []
            for idx1, city1 in enumerate(cities_copy):
                for city2 in cities_copy[idx1+1:]:  # Seules les villes qui suivent dans l'ordre topologique
                    if not nx.has_path(dag, city2, city1):  # Vérifier que l'ajout ne crée pas de cycle
                        valid_pairs.append((city1, city2))
            
            if not valid_pairs:
                break  # Arrêter si nous ne pouvons plus ajouter de dépendances sans créer de cycle
            
            # Choisir une paire valide et l'ajouter
            city1, city2 = random.choice(valid_pairs)
            dag.add_edge(city1, city2)
            dependencies.append((city1, city2))
        
        # Ajouter les dépendances comme attribut du graphe
        self.graph.graph['dependencies'] = dependencies
        
        return dependencies
    
    def block_routes(self, num_blocked: Optional[int] = None, 
                   blocking_ratio: float = 0.1) -> List[Tuple[str, str]]:
        """
        Bloque aléatoirement certaines routes entre villes.
        
        Args:
            num_blocked: Nombre de routes à bloquer, si None, utilise blocking_ratio
            blocking_ratio: Ratio du nombre de routes bloquées par rapport au nombre total de routes
        
        Returns:
            Liste des routes bloquées sous forme de tuples (ville1, ville2)
        """
        if not self.graph:
            raise ValueError("Il faut d'abord générer un graphe avec generate_complete_graph()")
        
        # Liste des arêtes disponibles
        edges = list(self.graph.edges())
        
        # Déterminer le nombre de routes à bloquer
        if num_blocked is None:
            num_blocked = int(blocking_ratio * len(edges))
        
        # S'assurer qu'on ne bloque pas trop de routes pour garder le graphe connexe
        max_block = len(edges) - (len(self.cities) - 1)  # Nombre maximum pour garder le graphe connexe
        num_blocked = min(num_blocked, max_block - 1)  # Garder une marge de sécurité
        
        if num_blocked <= 0:
            return []  # Pas de routes à bloquer
        
        # Mélanger la liste pour choisir des routes aléatoirement
        random.shuffle(edges)
        
        blocked_routes = []
        test_graph = self.graph.copy()
        
        for i in range(min(num_blocked, len(edges))):
            edge = edges[i]
            
            # Supprimer temporairement cette arête et vérifier si le graphe reste connexe
            test_graph.remove_edge(*edge)
            
            if nx.is_connected(test_graph):
                # Accepter ce blocage
                blocked_routes.append(edge)
            else:
                # Restaurer l'arête, car sa suppression déconnecte le graphe
                test_graph.add_edge(*edge, weight=self.graph[edge[0]][edge[1]]['weight'])
        
        # Ajouter les routes bloquées comme attribut du graphe
        self.graph.graph['blocked_routes'] = blocked_routes
        
        # Mettre à jour le graphe en supprimant les routes bloquées
        for city1, city2 in blocked_routes:
            self.graph.remove_edge(city1, city2)
        
        return blocked_routes
    
    def generate_tsp_graph(self, num_cities: int, geographical: bool = True,
                          min_load: int = 10, max_load: int = 100, 
                          dependency_ratio: float = 0.2, blocking_ratio: float = 0.1,
                          seed: Optional[int] = None) -> nx.Graph:
        """
        Génère un graphe complet pour le TSP avec toutes les contraintes hybrides.
        
        Args:
            num_cities: Nombre de villes
            geographical: Si True, utilise des distances géographiques
            min_load: Charge minimale pour une ville
            max_load: Charge maximale pour une ville
            dependency_ratio: Ratio de dépendances à ajouter
            blocking_ratio: Ratio de routes à bloquer
            seed: Valeur d'initialisation pour la génération aléatoire
            
        Returns:
            Graphe NetworkX configuré pour le TSP avec toutes les contraintes
        """
        # Réinitialiser le générateur aléatoire si un seed est fourni
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.__init__(seed)
        
        # Générer le graphe de base
        self.generate_complete_graph(num_cities, geographical=geographical)
        
        # Ajouter les charges des villes
        self.add_city_loads(min_load, max_load)
        
        # Ajouter les dépendances
        self.add_dependencies(dependency_ratio=dependency_ratio)
        
        # Bloquer certaines routes
        self.block_routes(blocking_ratio=blocking_ratio)
        
        return self.graph
    
    def export_graph_data(self) -> Dict:
        """
        Exporte les données du graphe sous forme de dictionnaire.
        
        Returns:
            Dictionnaire contenant toutes les informations du graphe
        """
        if not self.graph:
            raise ValueError("Il faut d'abord générer un graphe")
        
        # Extraire les données du graphe
        data = {
            "num_cities": len(self.cities),
            "cities": self.cities,
            "edges": [{
                "from": u, 
                "to": v, 
                "weight": self.graph[u][v]['weight']
            } for u, v in self.graph.edges()],
            "loads": {city: self.graph.nodes[city].get('load', 0) for city in self.cities},
        }
        
        # Ajouter les contraintes s'il y en a
        if 'dependencies' in self.graph.graph:
            data['dependencies'] = self.graph.graph['dependencies']
        
        if 'blocked_routes' in self.graph.graph:
            data['blocked_routes'] = self.graph.graph['blocked_routes']
        
        return data


# Fonction utilitaire pour tester le générateur
def create_test_graph(num_cities=10, seed=None):
    """
    Crée un graphe de test avec les paramètres par défaut.
    
    Args:
        num_cities: Nombre de villes dans le graphe
        seed: Valeur d'initialisation pour la génération aléatoire
        
    Returns:
        Un graphe TSP avec contraintes hybrides
    """
    generator = TSPGraphGenerator(seed=seed)
    graph = generator.generate_tsp_graph(
        num_cities=num_cities,
        geographical=True,
        min_load=10,
        max_load=100,
        dependency_ratio=0.2,
        blocking_ratio=0.1
    )
    
    print(f"Graphe généré avec {len(generator.cities)} villes")
    print(f"Nombre d'arêtes: {graph.number_of_edges()}")
    if 'dependencies' in graph.graph:
        print(f"Nombre de dépendances: {len(graph.graph['dependencies'])}")
    if 'blocked_routes' in graph.graph:
        print(f"Nombre de routes bloquées: {len(graph.graph['blocked_routes'])}")
    
    return graph, generator.export_graph_data()


if __name__ == "__main__":
    # Test du générateur
    graph, data = create_test_graph(num_cities=15, seed=42)
    print("\nDonnées exportées du graphe:")
    print(f"Villes: {data['cities']}")
    print(f"Charges des villes: {data['loads']}")
    if 'dependencies' in data:
        print(f"Dépendances: {data['dependencies']}")
    if 'blocked_routes' in data:
        print(f"Routes bloquées: {data['blocked_routes']}")