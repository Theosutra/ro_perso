"""
results_logger.py

Ce module gère l'enregistrement et l'export des résultats d'exécution
de l'algorithme génétique pour le TSP avec contraintes hybrides.
"""

import json
import csv
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class ResultsLogger:
    """
    Classe pour enregistrer et exporter les résultats des exécutions d'algorithmes.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialise le logger de résultats.
        
        Args:
            output_dir: Répertoire de sortie pour les fichiers de résultats
        """
        self.output_dir = output_dir
        self.results = []
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Ajoute un résultat à la liste des résultats.
        
        Args:
            result: Dictionnaire contenant les informations du résultat
        """
        # Ajouter un horodatage si non présent
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        
        self.results.append(result)
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Enregistre tous les résultats dans un fichier JSON.
        
        Args:
            filename: Nom du fichier de sortie (optionnel)
            
        Returns:
            Chemin complet du fichier enregistré
        """
        if filename is None:
            # Générer un nom de fichier avec horodatage
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"tsp_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Résultats enregistrés dans {filepath}")
        return filepath
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Enregistre les résultats principaux dans un fichier CSV.
        
        Args:
            filename: Nom du fichier de sortie (optionnel)
            
        Returns:
            Chemin complet du fichier enregistré
        """
        if not self.results:
            print("Aucun résultat à enregistrer")
            return ""
        
        if filename is None:
            # Générer un nom de fichier avec horodatage
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"tsp_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Identifier les champs à enregistrer (premiers niveaux du dictionnaire)
        # et les champs imbriqués importants
        first_result = self.results[0]
        
        # Préparer les en-têtes du CSV
        headers = ['execution_id', 'solver', 'timestamp']
        
        # Ajouter les paramètres d'entrée
        if 'input_parameters' in first_result:
            for param_key in first_result['input_parameters']:
                headers.append(f"param_{param_key}")
        
        # Ajouter les métriques de performance
        if 'performance_metrics' in first_result:
            for metric_key in first_result['performance_metrics']:
                headers.append(f"perf_{metric_key}")
        
        # Ajouter les métriques de contraintes
        if 'constraints' in first_result:
            for const_key in first_result['constraints']:
                headers.append(f"const_{const_key}")
        
        # Ajouter les statistiques de chargement
        if 'vehicle_load_stats' in first_result:
            for load_key in first_result['vehicle_load_stats']:
                if load_key != 'load_distribution':  # Éviter les listes
                    headers.append(f"load_{load_key}")
        
        # Ajouter le coût total
        headers.append('total_cost')
        
        # Écrire le fichier CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for result in self.results:
                row = {}
                
                # Ajouter les champs de base
                row['execution_id'] = result.get('execution_id', '')
                row['solver'] = result.get('solver', '')
                row['timestamp'] = result.get('timestamp', '')
                row['total_cost'] = result.get('total_cost', '')
                
                # Ajouter les paramètres d'entrée
                if 'input_parameters' in result:
                    for param_key, param_value in result['input_parameters'].items():
                        row[f"param_{param_key}"] = param_value
                
                # Ajouter les métriques de performance
                if 'performance_metrics' in result:
                    for metric_key, metric_value in result['performance_metrics'].items():
                        row[f"perf_{metric_key}"] = metric_value
                
                # Ajouter les métriques de contraintes
                if 'constraints' in result:
                    for const_key, const_value in result['constraints'].items():
                        row[f"const_{const_key}"] = const_value
                
                # Ajouter les statistiques de chargement
                if 'vehicle_load_stats' in result:
                    for load_key, load_value in result['vehicle_load_stats'].items():
                        if load_key != 'load_distribution':  # Éviter les listes
                            row[f"load_{load_key}"] = load_value
                
                writer.writerow(row)
        
        print(f"Résultats CSV enregistrés dans {filepath}")
        return filepath
    
    def save_solution_path(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Enregistre le chemin solution dans un fichier texte séparé.
        
        Args:
            result: Dictionnaire contenant les informations du résultat
            filename: Nom du fichier de sortie (optionnel)
            
        Returns:
            Chemin complet du fichier enregistré
        """
        if 'solution_path' not in result:
            print("Aucun chemin solution dans le résultat")
            return ""
        
        if filename is None:
            # Générer un nom de fichier avec ID d'exécution
            exec_id = result.get('execution_id', str(uuid.uuid4())[:8])
            filename = f"solution_path_{exec_id}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Écrire les informations de base
            f.write(f"Chemin solution pour l'exécution: {result.get('execution_id', '')}\n")
            f.write(f"Timestamp: {result.get('timestamp', '')}\n")
            f.write(f"Coût total: {result.get('total_cost', '')}\n")
            f.write(f"Nombre de villes: {result.get('input_parameters', {}).get('num_cities', '')}\n\n")
            
            # Écrire le chemin solution
            f.write("Chemin solution:\n")
            path = result['solution_path']
            
            for i, city in enumerate(path):
                f.write(f"{i+1}. {city}")
                
                # Ajouter la charge si disponible
                load_distribution = result.get('vehicle_load_stats', {}).get('load_distribution', [])
                if i < len(load_distribution):
                    f.write(f" (Charge: {load_distribution[i]})")
                
                f.write("\n")
            
            # Fermer la boucle
            f.write(f"{len(path)+1}. {path[0]} (retour au point de départ)\n")
            
            # Écrire les statistiques de charge
            if 'vehicle_load_stats' in result:
                f.write("\nStatistiques de charge:\n")
                stats = result['vehicle_load_stats']
                f.write(f"Capacité du véhicule: {result.get('input_parameters', {}).get('vehicle_capacity', '')}\n")
                f.write(f"Charge totale livrée: {stats.get('total_volume_delivered', '')}\n")
                f.write(f"Charge maximale atteinte: {stats.get('maximum_load', '')}\n")
                f.write(f"Capacité restante: {stats.get('remaining_capacity', '')}\n")
            
            # Écrire les informations de performance
            if 'performance_metrics' in result:
                f.write("\nPerformance:\n")
                perf = result['performance_metrics']
                f.write(f"Temps d'exécution: {perf.get('execution_time_sec', '')} secondes\n")
                f.write(f"Nombre d'itérations: {perf.get('iterations', '')}\n")
        
        print(f"Chemin solution enregistré dans {filepath}")
        return filepath
    
    def compare_results(self, metric_key: str = 'total_cost') -> None:
        """
        Compare les résultats sur une métrique donnée et affiche un résumé.
        
        Args:
            metric_key: Clé de la métrique à comparer
        """
        if not self.results:
            print("Aucun résultat à comparer")
            return
        
        print(f"\nComparaison des résultats sur la métrique: {metric_key}")
        print("-" * 80)
        
        # Trier les résultats par la métrique
        sorted_results = sorted(self.results, key=lambda r: r.get(metric_key, float('inf')))
        
        # Afficher les meilleurs résultats
        print(f"Top 5 des meilleurs résultats:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"#{i+1}: {metric_key}={result.get(metric_key, 'N/A')} - "
                  f"Exécution: {result.get('execution_id', 'N/A')} - "
                  f"Solver: {result.get('solver', 'N/A')}")
            
            # Afficher les paramètres utilisés
            if 'input_parameters' in result:
                params = result['input_parameters']
                print(f"    Paramètres: {', '.join([f'{k}={v}' for k, v in params.items()])}")
            
            # Afficher si les contraintes sont respectées
            if 'constraints' in result:
                constraints = result['constraints']
                const_str = ', '.join([f'{k}={v}' for k, v in constraints.items()])
                print(f"    Contraintes: {const_str}")
            
            print()
        
        # Calculer quelques statistiques
        metric_values = [r.get(metric_key, float('inf')) for r in self.results if metric_key in r]
        if metric_values:
            avg_value = sum(metric_values) / len(metric_values)
            min_value = min(metric_values)
            max_value = max(metric_values)
            
            print(f"Statistiques pour {metric_key}:")
            print(f"    Minimum: {min_value}")
            print(f"    Maximum: {max_value}")
            print(f"    Moyenne: {avg_value:.2f}")
            print(f"    Nombre de résultats: {len(metric_values)}")
    
    def clear(self) -> None:
        """Efface tous les résultats stockés."""
        self.results = []
        print("Tous les résultats ont été effacés de la mémoire.")


# Fonction utilitaire pour tester le logger
def test_logger():
    """
    Fonction de test pour le logger de résultats.
    """
    logger = ResultsLogger(output_dir="test_results")
    
    # Créer quelques résultats de test
    for i in range(3):
        result = {
            "execution_id": str(uuid.uuid4()),
            "solver": "GeneticAlgorithm",
            "timestamp": datetime.now().isoformat(),
            "input_parameters": {
                "num_cities": 10 + i * 5,
                "vehicle_capacity": 1000,
                "mutation_rate": 0.05 + i * 0.02,
                "population_size": 100 + i * 50
            },
            "solution_path": ["A", "C", "E", "B", "D"],
            "total_cost": 250 - i * 10,
            "performance_metrics": {
                "execution_time_sec": 1.5 + i * 0.5,
                "iterations": 200 + i * 50
            },
            "constraints": {
                "dependencies_respected": True,
                "capacity_respected": i < 2  # Le dernier viole la contrainte
            },
            "vehicle_load_stats": {
                "total_volume_delivered": 800 + i * 100,
                "maximum_load": 900 + i * 50,
                "remaining_capacity": 100 - i * 50,
                "load_distribution": [100, 200, 300, 150, 50]
            }
        }
        
        logger.add_result(result)
    
    # Tester les différentes méthodes d'exportation
    logger.save_to_json()
    logger.save_to_csv()
    logger.save_solution_path(logger.results[0])
    
    # Comparer les résultats
    logger.compare_results()


if __name__ == "__main__":
    # Tester le logger
    test_logger()