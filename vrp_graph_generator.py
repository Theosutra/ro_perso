"""
vrp_graph_generator.py

Ce module permet de générer des graphes aléatoires pour tester l'algorithme VRP
avec contraintes hybrides (dépendances, routes bloquées, flottes de véhicules
et contraintes de chargement).
"""

import networkx as nx
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Set, Optional, Union
from graph_generator import TSPGraphGenerator


class VRPGraphGenerator:
    """
    Générateur de graphes pour le problème VRP avec contraintes hybrides.
    Étend le générateur TSP avec des fonctionnalités spécifiques au VRP.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialise le générateur de graphes VRP.
        
        Args:
            seed: Valeur d'initialisation pour la génération aléatoire (reproductibilité)
        """
        # Utiliser le générateur TSP comme base
        self.tsp_generator = TSPGraphGenerator(seed=seed)
        
        # Variables internes
        self.graph = None
        self.cities = []
        self.depot = None
        self.vehicles = []
        
        # Initialiser le générateur aléatoire
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_vrp_graph(self, num_cities: int, num_vehicles: int = 3, 
                         depot_name: Optional[str] = None,
                         min_vehicle_capacity: int = 500, max_vehicle_capacity: int = 1500,
                         homogeneous_fleet: bool = True,
                         geographical: bool = True,
                         min_load: int = 10, max_load: int = 100, 
                         dependency_ratio: float = 0.2, blocking_ratio: float = 0.1,
                         time_windows: bool = False,
                         seed: Optional[int] = None) -> nx.Graph:
        """
        Génère un graphe complet pour le VRP avec toutes les contraintes hybrides.
        
        Args:
            num_cities: Nombre de villes (sans compter le dépôt)
            num_vehicles: Nombre de véhicules dans la flotte
            depot_name: Nom du dépôt (si None, 'Depot' sera utilisé)
            min_vehicle_capacity: Capacité minimale d'un véhicule
            max_vehicle_capacity: Capacité maximale d'un véhicule
            homogeneous_fleet: Si True, tous les véhicules ont la même capacité
            geographical: Si True, utilise des distances géographiques
            min_load: Charge minimale pour une ville
            max_load: Charge maximale pour une ville
            dependency_ratio: Ratio de dépendances à ajouter
            blocking_ratio: Ratio de routes à bloquer
            time_windows: Si True, ajoute des fenêtres temporelles aux villes
            seed: Valeur d'initialisation pour la génération aléatoire
            
        Returns:
            Graphe NetworkX configuré pour le VRP avec toutes les contraintes
        """
        # Réinitialiser le générateur aléatoire si un seed est fourni
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.tsp_generator.__init__(seed)
        
        # Générer le nom du dépôt s'il n'est pas fourni
        if depot_name is None:
            depot_name = "Depot"
        self.depot = depot_name
        
        # Générer un graphe TSP de base (sans le dépôt pour l'instant)
        tsp_graph = self.tsp_generator.generate_complete_graph(
            num_cities=num_cities, 
            geographical=geographical
        )
        
        # Ajouter les charges aux villes
        self.tsp_generator.add_city_loads(min_load, max_load)
        
        # Récupérer les villes du graphe TSP
        client_cities = list(self.tsp_generator.cities)
        
        # Créer un nouveau graphe VRP
        self.graph = nx.Graph()
        
        # Ajouter le dépôt
        self.graph.add_node(self.depot, load=0, is_depot=True)
        
        # Ajouter les villes clients
        for city in client_cities:
            city_data = tsp_graph.nodes[city]
            self.graph.add_node(city, **city_data, is_depot=False)
        
        # Mise à jour de la liste des villes
        self.cities = [self.depot] + client_cities
        
        # Ajouter les arêtes du graphe TSP
        for u, v, data in tsp_graph.edges(data=True):
            self.graph.add_edge(u, v, **data)
        
        # Connecter le dépôt à toutes les villes
        if geographical and 'pos' in self.graph.nodes[client_cities[0]]:
            # Positionner le dépôt au centre des villes pour les graphes géographiques
            positions = nx.get_node_attributes(self.graph, 'pos')
            avg_x = sum(pos[0] for pos in positions.values()) / len(positions)
            avg_y = sum(pos[1] for pos in positions.values()) / len(positions)
            self.graph.nodes[self.depot]['pos'] = (avg_x, avg_y)
            
            # Calculer les distances euclidiennes entre le dépôt et les villes
            for city in client_cities:
                x1, y1 = self.graph.nodes[self.depot]['pos']
                x2, y2 = self.graph.nodes[city]['pos']
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.graph.add_edge(self.depot, city, weight=distance)
        else:
            # Pour les graphes non géographiques, ajouter des distances aléatoires
            for city in client_cities:
                distance = random.uniform(10, 100)
                self.graph.add_edge(self.depot, city, weight=distance)
        
        # Ajouter les dépendances entre villes
        dependencies = []
        if dependency_ratio > 0:
            max_dependencies = int(dependency_ratio * len(client_cities))
            for _ in range(max_dependencies):
                # Sélectionner deux villes aléatoires (pas le dépôt)
                city1, city2 = random.sample(client_cities, 2)
                dependencies.append((city1, city2))
        
        # Stocker les dépendances dans le graphe
        self.graph.graph['dependencies'] = dependencies
        
        # Bloquer certaines routes si demandé
        blocked_routes = []
        if blocking_ratio > 0:
            edges = list(self.graph.edges())
            num_blocked = int(blocking_ratio * len(edges))
            
            # Éviter de bloquer des routes vers/depuis le dépôt
            valid_edges = [(u, v) for u, v in edges if u != self.depot and v != self.depot]
            
            # S'assurer qu'on ne bloque pas trop de routes pour garder le graphe connexe
            max_block = len(valid_edges) - len(client_cities)  # Nombre maximum pour garder le graphe connexe
            num_blocked = min(num_blocked, max_block - 1)  # Garder une marge de sécurité
            
            if num_blocked > 0:
                random.shuffle(valid_edges)
                
                test_graph = self.graph.copy()
                for i in range(min(num_blocked, len(valid_edges))):
                    edge = valid_edges[i]
                    
                    # Supprimer temporairement cette arête et vérifier si le graphe reste connexe
                    test_graph.remove_edge(*edge)
                    
                    if nx.is_connected(test_graph):
                        # Accepter ce blocage
                        blocked_routes.append(edge)
                    else:
                        # Restaurer l'arête, car sa suppression déconnecte le graphe
                        test_graph.add_edge(*edge, weight=self.graph[edge[0]][edge[1]]['weight'])
                
                # Supprimer les routes bloquées du graphe
                for u, v in blocked_routes:
                    self.graph.remove_edge(u, v)
        
        # Stocker les routes bloquées dans le graphe
        self.graph.graph['blocked_routes'] = blocked_routes
        
        # Générer la flotte de véhicules
        self.generate_vehicle_fleet(num_vehicles, min_vehicle_capacity, max_vehicle_capacity, homogeneous_fleet)
        
        # Ajouter les fenêtres temporelles si demandé
        if time_windows:
            self.add_time_windows()
        
        return self.graph
    
    def generate_vehicle_fleet(self, num_vehicles: int, min_capacity: int,
                              max_capacity: int, homogeneous: bool) -> List[Dict]:
        """
        Génère une flotte de véhicules avec des capacités.
        
        Args:
            num_vehicles: Nombre de véhicules dans la flotte
            min_capacity: Capacité minimale d'un véhicule
            max_capacity: Capacité maximale d'un véhicule
            homogeneous: Si True, tous les véhicules ont la même capacité
            
        Returns:
            Liste des véhicules avec leurs caractéristiques
        """
        vehicles = []
        
        if homogeneous:
            # Flotte homogène: tous les véhicules ont la même capacité
            capacity = random.randint(min_capacity, max_capacity)
            for i in range(num_vehicles):
                vehicles.append({
                    'id': i + 1,
                    'capacity': capacity,
                    'depot': self.depot
                })
        else:
            # Flotte hétérogène: capacités différentes
            for i in range(num_vehicles):
                capacity = random.randint(min_capacity, max_capacity)
                vehicles.append({
                    'id': i + 1,
                    'capacity': capacity,
                    'depot': self.depot
                })
        
        # Stocker la flotte dans le graphe
        self.graph.graph['vehicles'] = vehicles
        self.vehicles = vehicles
        
        return vehicles
    
    def add_time_windows(self, day_length: int = 480) -> Dict[str, Tuple[int, int]]:
        """
        Ajoute des fenêtres temporelles pour chaque ville.
        La journée est divisée en minutes, par défaut 8 heures = 480 minutes.
        
        Args:
            day_length: Longueur de la journée en minutes
            
        Returns:
            Dictionnaire des fenêtres temporelles par ville
        """
        time_windows = {}
        
        # Le dépôt est disponible toute la journée
        time_windows[self.depot] = (0, day_length)
        
        # Générer des fenêtres temporelles aléatoires pour les villes clientes
        for city in self.cities:
            if city != self.depot:
                # Générer une fenêtre d'une largeur aléatoire entre 60 et 180 minutes
                window_width = random.randint(60, 180)
                # Heure de début entre 0 et (day_length - window_width)
                earliest = random.randint(0, day_length - window_width)
                latest = earliest + window_width
                
                time_windows[city] = (earliest, latest)
        
        # Stocker les fenêtres temporelles dans le graphe
        nx.set_node_attributes(self.graph, time_windows, 'time_window')
        self.graph.graph['time_windows'] = time_windows
        
        return time_windows
    
    def add_service_times(self, min_service: int = 10, max_service: int = 30) -> Dict[str, int]:
        """
        Ajoute des temps de service pour chaque ville (temps nécessaire pour le service sur place).
        
        Args:
            min_service: Temps minimum de service en minutes
            max_service: Temps maximum de service en minutes
            
        Returns:
            Dictionnaire des temps de service par ville
        """
        service_times = {}
        
        # Le service au dépôt est minimal (chargement/déchargement)
        service_times[self.depot] = min_service
        
        # Générer des temps de service pour les villes clientes
        for city in self.cities:
            if city != self.depot:
                service_times[city] = random.randint(min_service, max_service)
        
        # Stocker les temps de service dans le graphe
        nx.set_node_attributes(self.graph, service_times, 'service_time')
        self.graph.graph['service_times'] = service_times
        
        return service_times
    
    def add_priority_levels(self, num_levels: int = 3) -> Dict[str, int]:
        """
        Ajoute des niveaux de priorité aux villes (1 = haute priorité, num_levels = basse priorité).
        
        Args:
            num_levels: Nombre de niveaux de priorité
            
        Returns:
            Dictionnaire des niveaux de priorité par ville
        """
        priorities = {}
        
        # Le dépôt n'a pas de priorité
        priorities[self.depot] = 0
        
        # Générer des priorités pour les villes clientes
        for city in self.cities:
            if city != self.depot:
                priorities[city] = random.randint(1, num_levels)
        
        # Stocker les priorités dans le graphe
        nx.set_node_attributes(self.graph, priorities, 'priority')
        self.graph.graph['priorities'] = priorities
        
        return priorities
    
    def export_graph_data(self) -> Dict:
        """
        Exporte les données du graphe VRP sous forme de dictionnaire.
        
        Returns:
            Dictionnaire contenant toutes les informations du graphe
        """
        if not self.graph:
            raise ValueError("Il faut d'abord générer un graphe")
        
        # Extraire les données du graphe
        data = {
            "num_cities": len(self.cities) - 1,  # Ne pas compter le dépôt
            "depot": self.depot,
            "cities": self.cities,
            "vehicles": self.vehicles,
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
        
        if 'time_windows' in self.graph.graph:
            data['time_windows'] = self.graph.graph['time_windows']
        
        if 'service_times' in self.graph.graph:
            data['service_times'] = self.graph.graph['service_times']
        
        if 'priorities' in self.graph.graph:
            data['priorities'] = self.graph.graph['priorities']
        
        return data


# Fonction utilitaire pour tester le générateur
def create_test_vrp_graph(num_cities=10, num_vehicles=3, 
                         with_time_windows=False, homogeneous_fleet=True, seed=None):
    """
    Crée un graphe VRP de test avec les paramètres fournis.
    
    Args:
        num_cities: Nombre de villes dans le graphe (hors dépôt)
        num_vehicles: Nombre de véhicules dans la flotte
        with_time_windows: Si True, ajoute des fenêtres temporelles
        homogeneous_fleet: Si True, tous les véhicules ont la même capacité
        seed: Valeur d'initialisation pour la génération aléatoire
        
    Returns:
        Un tuple (graphe VRP, données exportées)
    """
    generator = VRPGraphGenerator(seed=seed)
    graph = generator.generate_vrp_graph(
        num_cities=num_cities,
        num_vehicles=num_vehicles,
        geographical=True,
        min_load=10,
        max_load=100,
        dependency_ratio=0.2,
        blocking_ratio=0.1,
        time_windows=with_time_windows,
        homogeneous_fleet=homogeneous_fleet
    )
    
    if with_time_windows:
        generator.add_service_times()
    
    print(f"Graphe VRP généré avec {len(generator.cities)-1} villes et 1 dépôt")
    print(f"Nombre d'arêtes: {graph.number_of_edges()}")
    print(f"Nombre de véhicules: {len(generator.vehicles)}")
    
    if 'dependencies' in graph.graph:
        print(f"Nombre de dépendances: {len(graph.graph['dependencies'])}")
    
    if 'blocked_routes' in graph.graph:
        print(f"Nombre de routes bloquées: {len(graph.graph['blocked_routes'])}")
    
    if 'time_windows' in graph.graph:
        print(f"Fenêtres temporelles activées")
    
    return graph, generator.export_graph_data()


if __name__ == "__main__":
    # Test du générateur
    graph, data = create_test_vrp_graph(
        num_cities=15, 
        num_vehicles=4, 
        with_time_windows=True, 
        homogeneous_fleet=False,
        seed=42
    )
    
    print("\nDonnées exportées du graphe:")
    print(f"Dépôt: {data['depot']}")
    print(f"Villes: {data['cities']}")
    print(f"Véhicules: {data['vehicles']}")
    
    if 'time_windows' in data:
        print("\nExemples de fenêtres temporelles:")
        for city, window in list(data['time_windows'].items())[:5]:
            print(f"  {city}: {window[0]}-{window[1]} minutes")