"""
analyze_benchmark.py

Script pour analyser les résultats des benchmarks et générer des visualisations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from datetime import datetime


def load_benchmark_results(file_path):
    """
    Charge les résultats d'un benchmark à partir d'un fichier CSV.
    
    Args:
        file_path: Chemin vers le fichier CSV de résultats
        
    Returns:
        DataFrame pandas contenant les résultats
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    df = pd.read_csv(file_path)
    print(f"Chargé {len(df)} résultats depuis {file_path}")
    return df


def create_analysis_directory(base_dir='benchmark_analysis'):
    """
    Crée un répertoire pour stocker les analyses et visualisations.
    
    Args:
        base_dir: Répertoire de base
        
    Returns:
        Chemin du répertoire créé
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    return analysis_dir


def analyze_by_city_size(df, output_dir):
    """
    Analyse l'impact de la taille des villes sur différentes métriques.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les visualisations
    """
    print("\nAnalyse de l'impact de la taille des villes...")
    
    # Créer un subplot avec plusieurs métriques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact de la taille des villes sur les performances', fontsize=16)
    
    # Coût total vs taille des villes
    sns.boxplot(x='num_cities', y='total_cost', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Coût total par taille de ville')
    axes[0, 0].set_xlabel('Nombre de villes')
    axes[0, 0].set_ylabel('Coût total')
    
    # Temps d'exécution vs taille des villes
    sns.boxplot(x='num_cities', y='execution_time', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Temps d\'exécution par taille de ville')
    axes[0, 1].set_xlabel('Nombre de villes')
    axes[0, 1].set_ylabel('Temps d\'exécution (s)')
    
    # Respect des contraintes vs taille des villes
    city_sizes = sorted(df['num_cities'].unique())
    
    # Taux de respect des dépendances
    dep_respect_rates = [df[df['num_cities'] == size]['dependencies_respected'].mean() * 100 
                        for size in city_sizes]
    
    # Taux de respect de la capacité
    cap_respect_rates = [df[df['num_cities'] == size]['capacity_respected'].mean() * 100 
                        for size in city_sizes]
    
    axes[1, 0].bar(np.array(range(len(city_sizes))) - 0.2, dep_respect_rates, width=0.4, label='Dépendances')
    axes[1, 0].bar(np.array(range(len(city_sizes))) + 0.2, cap_respect_rates, width=0.4, label='Capacité')
    axes[1, 0].set_xticks(range(len(city_sizes)))
    axes[1, 0].set_xticklabels(city_sizes)
    axes[1, 0].set_title('Taux de respect des contraintes par taille de ville')
    axes[1, 0].set_xlabel('Nombre de villes')
    axes[1, 0].set_ylabel('Taux de respect (%)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 100)
    
    # Nombre d'itérations vs taille des villes
    sns.boxplot(x='num_cities', y='iterations', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Nombre d\'itérations par taille de ville')
    axes[1, 1].set_xlabel('Nombre de villes')
    axes[1, 1].set_ylabel('Nombre d\'itérations')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'city_size_analysis.png'), dpi=300)
    plt.close()


def analyze_capacity_constraints(df, output_dir):
    """
    Analyse l'impact de la capacité du véhicule sur le respect des contraintes.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les visualisations
    """
    print("\nAnalyse des contraintes de capacité...")
    
    # Créer un subplot avec plusieurs métriques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse des contraintes de capacité', fontsize=16)
    
    # Respect de la capacité vs capacité du véhicule
    capacities = sorted(df['vehicle_capacity'].unique())
    capacity_respect_rates = [df[df['vehicle_capacity'] == cap]['capacity_respected'].mean() * 100 
                             for cap in capacities]
    
    axes[0, 0].bar(capacities, capacity_respect_rates)
    axes[0, 0].set_title('Taux de respect de la capacité vs capacité du véhicule')
    axes[0, 0].set_xlabel('Capacité du véhicule')
    axes[0, 0].set_ylabel('Taux de respect (%)')
    axes[0, 0].set_ylim(0, 100)
    
    # Relation entre charge totale et capacité
    sns.scatterplot(x='total_load', y='capacity_respected', 
                   hue='vehicle_capacity', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Respect de la capacité vs charge totale')
    axes[0, 1].set_xlabel('Charge totale')
    axes[0, 1].set_ylabel('Capacité respectée (1=Oui, 0=Non)')
    
    # Charge maximale vs capacité du véhicule pour les solutions réussies
    successful_df = df[df['capacity_respected'] == True]
    if not successful_df.empty:
        sns.boxplot(x='vehicle_capacity', y='max_load', data=successful_df, ax=axes[1, 0])
        axes[1, 0].set_title('Charge maximale pour les solutions respectant la capacité')
        axes[1, 0].set_xlabel('Capacité du véhicule')
        axes[1, 0].set_ylabel('Charge maximale')
    else:
        axes[1, 0].text(0.5, 0.5, 'Aucune solution ne respecte la contrainte de capacité', 
                       ha='center', va='center')
    
    # Ratio de charge (max_load/vehicle_capacity) vs nombre de villes
    df['load_ratio'] = df['max_load'] / df['vehicle_capacity']
    sns.boxplot(x='num_cities', y='load_ratio', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Ratio de charge (max/capacité) vs nombre de villes')
    axes[1, 1].set_xlabel('Nombre de villes')
    axes[1, 1].set_ylabel('Ratio de charge')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'capacity_analysis.png'), dpi=300)
    plt.close()


def analyze_algorithm_parameters(df, output_dir):
    """
    Analyse l'impact des paramètres de l'algorithme génétique.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les visualisations
    """
    print("\nAnalyse des paramètres de l'algorithme génétique...")
    
    # Créer un subplot avec plusieurs métriques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact des paramètres de l\'algorithme génétique', fontsize=16)
    
    # Impact de la taille de population sur le coût
    sns.boxplot(x='population_size', y='total_cost', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Coût total vs taille de la population')
    axes[0, 0].set_xlabel('Taille de la population')
    axes[0, 0].set_ylabel('Coût total')
    
    # Impact du taux de mutation sur le coût
    sns.boxplot(x='mutation_rate', y='total_cost', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Coût total vs taux de mutation')
    axes[0, 1].set_xlabel('Taux de mutation')
    axes[0, 1].set_ylabel('Coût total')
    
    # Impact de la taille de population sur le respect des contraintes
    pop_sizes = sorted(df['population_size'].unique())
    
    # Taux de respect des dépendances
    dep_respect_rates = [df[df['population_size'] == size]['dependencies_respected'].mean() * 100 
                        for size in pop_sizes]
    
    # Taux de respect de la capacité
    cap_respect_rates = [df[df['population_size'] == size]['capacity_respected'].mean() * 100 
                        for size in pop_sizes]
    
    axes[1, 0].bar(np.array(range(len(pop_sizes))) - 0.2, dep_respect_rates, width=0.4, label='Dépendances')
    axes[1, 0].bar(np.array(range(len(pop_sizes))) + 0.2, cap_respect_rates, width=0.4, label='Capacité')
    axes[1, 0].set_xticks(range(len(pop_sizes)))
    axes[1, 0].set_xticklabels(pop_sizes)
    axes[1, 0].set_title('Taux de respect des contraintes vs taille de population')
    axes[1, 0].set_xlabel('Taille de la population')
    axes[1, 0].set_ylabel('Taux de respect (%)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 100)
    
    # Impact du taux de mutation sur le nombre d'itérations
    sns.boxplot(x='mutation_rate', y='iterations', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Nombre d\'itérations vs taux de mutation')
    axes[1, 1].set_xlabel('Taux de mutation')
    axes[1, 1].set_ylabel('Nombre d\'itérations')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'algorithm_parameters_analysis.png'), dpi=300)
    plt.close()


def analyze_dependencies_and_blocking(df, output_dir):
    """
    Analyse l'impact des contraintes de dépendance et de blocage.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les visualisations
    """
    print("\nAnalyse des contraintes de dépendance et de blocage...")
    
    # Créer un subplot avec plusieurs métriques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact des contraintes de dépendance et de blocage', fontsize=16)
    
    # Impact du ratio de dépendance sur le coût
    sns.boxplot(x='dependency_ratio', y='total_cost', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Coût total vs ratio de dépendance')
    axes[0, 0].set_xlabel('Ratio de dépendance')
    axes[0, 0].set_ylabel('Coût total')
    
    # Impact du ratio de blocage sur le coût
    sns.boxplot(x='blocking_ratio', y='total_cost', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Coût total vs ratio de blocage')
    axes[0, 1].set_xlabel('Ratio de blocage')
    axes[0, 1].set_ylabel('Coût total')
    
    # Impact combiné des ratios de dépendance et de blocage sur le respect des contraintes
    heatmap_data = pd.pivot_table(
        df, 
        values='dependencies_respected', 
        index='dependency_ratio',
        columns='blocking_ratio',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1, 0])
    axes[1, 0].set_title('Taux de respect des dépendances')
    axes[1, 0].set_xlabel('Ratio de blocage')
    axes[1, 0].set_ylabel('Ratio de dépendance')
    
    # Impact combiné sur le temps d'exécution
    heatmap_data = pd.pivot_table(
        df, 
        values='execution_time', 
        index='dependency_ratio',
        columns='blocking_ratio',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Temps d\'exécution moyen (s)')
    axes[1, 1].set_xlabel('Ratio de blocage')
    axes[1, 1].set_ylabel('Ratio de dépendance')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'dependencies_blocking_analysis.png'), dpi=300)
    plt.close()


def generate_summary_statistics(df, output_dir):
    """
    Génère des statistiques résumées sur les résultats et les exporte en CSV.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les fichiers
    """
    print("\nGénération des statistiques résumées...")
    
    # Statistiques générales
    general_stats = {
        'total_runs': len(df),
        'avg_cost': df['total_cost'].mean(),
        'min_cost': df['total_cost'].min(),
        'max_cost': df['total_cost'].max(),
        'avg_execution_time': df['execution_time'].mean(),
        'avg_iterations': df['iterations'].mean(),
        'dependencies_respected_rate': df['dependencies_respected'].mean() * 100,
        'capacity_respected_rate': df['capacity_respected'].mean() * 100
    }
    
    # Créer un DataFrame pour ces statistiques
    stats_df = pd.DataFrame([general_stats])
    stats_df.to_csv(os.path.join(output_dir, 'general_statistics.csv'), index=False)
    
    # Statistiques par taille de ville
    city_stats = df.groupby('num_cities').agg({
        'total_cost': ['mean', 'min', 'max', 'std'],
        'execution_time': ['mean', 'min', 'max'],
        'iterations': ['mean', 'min', 'max'],
        'dependencies_respected': ['mean'],
        'capacity_respected': ['mean']
    })
    
    # Convertir les taux en pourcentages
    city_stats[('dependencies_respected', 'mean')] *= 100
    city_stats[('capacity_respected', 'mean')] *= 100
    
    # Renommer les colonnes pour plus de clarté
    city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
    
    # Exporter en CSV
    city_stats.to_csv(os.path.join(output_dir, 'city_size_statistics.csv'))
    
    # Statistiques par configuration d'algorithme
    algo_stats = df.groupby(['population_size', 'mutation_rate']).agg({
        'total_cost': ['mean', 'min', 'std'],
        'execution_time': ['mean'],
        'iterations': ['mean'],
        'dependencies_respected': ['mean'],
        'capacity_respected': ['mean']
    })
    
    # Convertir les taux en pourcentages
    algo_stats[('dependencies_respected', 'mean')] *= 100
    algo_stats[('capacity_respected', 'mean')] *= 100
    
    # Renommer les colonnes
    algo_stats.columns = ['_'.join(col).strip() for col in algo_stats.columns.values]
    
    # Exporter en CSV
    algo_stats.to_csv(os.path.join(output_dir, 'algorithm_statistics.csv'))
    
    # Statistiques sur les contraintes de capacité
    capacity_stats = df.groupby('vehicle_capacity').agg({
        'capacity_respected': ['mean'],
        'total_load': ['mean', 'min', 'max'],
        'max_load': ['mean', 'min', 'max']
    })
    
    # Convertir les taux en pourcentages
    capacity_stats[('capacity_respected', 'mean')] *= 100
    
    # Renommer les colonnes
    capacity_stats.columns = ['_'.join(col).strip() for col in capacity_stats.columns.values]
    
    # Exporter en CSV
    capacity_stats.to_csv(os.path.join(output_dir, 'capacity_statistics.csv'))
    
    print(f"Statistiques résumées exportées dans {output_dir}")


def find_best_solutions(df, output_dir):
    """
    Identifie et exporte les meilleures solutions trouvées.
    
    Args:
        df: DataFrame contenant les résultats
        output_dir: Répertoire de sortie pour les fichiers
    """
    print("\nRecherche des meilleures solutions...")
    
    # Filtrer les solutions qui respectent les deux contraintes
    valid_solutions = df[(df['dependencies_respected'] == True) & 
                        (df['capacity_respected'] == True)]
    
    if valid_solutions.empty:
        print("Aucune solution ne respecte toutes les contraintes.")
        # Prendre les meilleures solutions disponibles
        best_solutions = df.sort_values('total_cost').head(10)
    else:
        print(f"Trouvé {len(valid_solutions)} solutions valides qui respectent toutes les contraintes.")
        # Prendre les 10 meilleures solutions valides
        best_solutions = valid_solutions.sort_values('total_cost').head(10)
    
    # Exporter les meilleures solutions
    best_solutions.to_csv(os.path.join(output_dir, 'best_solutions.csv'), index=False)
    
    # Afficher un résumé des meilleures solutions
    print("\nTop 3 des meilleures solutions:")
    for i, row in best_solutions.head(3).iterrows():
        print(f"#{i+1}: Coût={row['total_cost']:.2f}, "
              f"Villes={row['num_cities']}, "
              f"Population={row['population_size']}, "
              f"Mutation={row['mutation_rate']}, "
              f"Dépendances={'Oui' if row['dependencies_respected'] else 'Non'}, "
              f"Capacité={'Oui' if row['capacity_respected'] else 'Non'}")
    
    return best_solutions


def run_analysis(csv_file, output_dir=None):
    """
    Exécute l'analyse complète des résultats de benchmark.
    
    Args:
        csv_file: Chemin vers le fichier CSV de résultats
        output_dir: Répertoire de sortie pour les analyses (optionnel)
    """
    # Charger les données
    df = load_benchmark_results(csv_file)
    
    # Créer le répertoire d'analyse si non spécifié
    if output_dir is None:
        output_dir = create_analysis_directory()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Analyse des résultats en cours, sortie dans {output_dir}...")
    
    # Effectuer les différentes analyses
    analyze_by_city_size(df, output_dir)
    analyze_capacity_constraints(df, output_dir)
    analyze_algorithm_parameters(df, output_dir)
    analyze_dependencies_and_blocking(df, output_dir)
    
    # Générer les statistiques résumées
    generate_summary_statistics(df, output_dir)
    
    # Trouver les meilleures solutions
    best_solutions = find_best_solutions(df, output_dir)
    
    print(f"\nAnalyse terminée. Tous les résultats sont disponibles dans {output_dir}")
    
    return df, best_solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse des résultats de benchmark TSP')
    
    parser.add_argument('csv_file', type=str, help='Fichier CSV contenant les résultats')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Répertoire de sortie pour l\'analyse (optionnel)')
    
    args = parser.parse_args()
    
    run_analysis(args.csv_file, args.output_dir)