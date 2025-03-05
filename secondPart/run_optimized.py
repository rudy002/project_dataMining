import os
import sys
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import product

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstPart.part1 import generate_data
from secondPart.part2 import bfr_cluster, cure_cluster
from secondPart.run_part2 import generate_test_data, cure_points_generator, calculate_cluster_accuracy

def optimize_parameters(algorithm_name, algorithm_func, data_path, tagged_path, dim, n, param_grid):
    """
    Optimise les paramètres d'un algorithme en testant différentes combinaisons.
    
    Args:
        algorithm_name: Nom de l'algorithme (BFR ou CURE)
        algorithm_func: Fonction d'algorithme à exécuter
        data_path: Chemin vers le fichier de données
        tagged_path: Chemin vers le fichier de données étiquetées
        dim: Dimension des données
        n: Nombre de points
        param_grid: Dictionnaire des paramètres à tester
    
    Returns:
        Tuple (meilleurs paramètres, meilleure précision, temps d'exécution)
    """
    print(f"\nOptimisation des paramètres pour {algorithm_name} sur {data_path}")
    
    # Générer toutes les combinaisons de paramètres
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_accuracy = 0
    best_params = None
    best_time = 0
    results = []
    
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_names, combination))
        print(f"\nTest {i+1}/{len(param_combinations)}: {params}")
        
        # Exécuter l'algorithme avec ces paramètres
        start_time = time.time()
        
        # Préparer les arguments
        kwargs = {
            'dim': dim,
            'n': n,
            'in_path': data_path,
            'out_path': f"temp_{algorithm_name.lower()}_result.csv"
        }
        kwargs.update(params)
        
        # Exécuter l'algorithme
        clusters = algorithm_func(**kwargs)
        
        execution_time = time.time() - start_time
        
        # Calculer la précision
        formatted_clusters = []
        for cluster in clusters:
            if 'points' in cluster:
                formatted_points = [tuple(p) if hasattr(p, '__iter__') else p for p in cluster['points']]
                formatted_clusters.append(formatted_points)
            else:
                formatted_points = [tuple(p) if hasattr(p, '__iter__') else p for p in cluster.get('points', [])]
                formatted_clusters.append(formatted_points)
        
        accuracy = calculate_cluster_accuracy(formatted_clusters, tagged_path, dim)
        
        print(f"Précision: {accuracy:.4f}, Temps: {execution_time:.2f}s")
        
        # Enregistrer les résultats
        result = {
            'params': params,
            'accuracy': accuracy,
            'time': execution_time
        }
        results.append(result)
        
        # Mettre à jour les meilleurs paramètres
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_time = execution_time
    
    # Nettoyer les fichiers temporaires
    if os.path.exists(f"temp_{algorithm_name.lower()}_result.csv"):
        os.remove(f"temp_{algorithm_name.lower()}_result.csv")
    
    print(f"\nMeilleurs paramètres pour {algorithm_name}:")
    print(f"Paramètres: {best_params}")
    print(f"Précision: {best_accuracy:.4f}")
    print(f"Temps: {best_time:.2f}s")
    
    return best_params, best_accuracy, best_time, results

def plot_results(results, param_name, algorithm_name, metric='accuracy'):
    """
    Trace un graphique montrant l'impact d'un paramètre sur la précision ou le temps.
    """
    # Extraire les valeurs uniques du paramètre
    param_values = sorted(set(result['params'][param_name] for result in results))
    
    # Calculer la moyenne pour chaque valeur de paramètre
    metric_values = []
    for value in param_values:
        matching_results = [r for r in results if r['params'][param_name] == value]
        avg_metric = sum(r[metric] for r in matching_results) / len(matching_results)
        metric_values.append(avg_metric)
    
    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, metric_values, 'o-', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('Précision' if metric == 'accuracy' else 'Temps (s)')
    plt.title(f"Impact de {param_name} sur {'la précision' if metric == 'accuracy' else 'le temps'} - {algorithm_name}")
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.savefig(f"secondPart/{algorithm_name.lower()}_{param_name}_{metric}.png")
    plt.close()

def main():
    # Créer les répertoires
    os.makedirs("BFR_data_test", exist_ok=True)
    os.makedirs("CURE_data_test", exist_ok=True)
    
    # Paramètres
    n = 10000        # Nombre de points (petit pour les tests)
    dim_bfr = 5      # Dimension pour BFR
    dim_cure = 6     # Dimension pour CURE
    k_values = [5, 10, 15, 20]  # Différentes valeurs de k à tester
    block_size_values = [500, 1000, 2000]  # Différentes tailles de bloc
    
    # Générer les données BFR si elles n'existent pas
    bfr_path = "BFR_data_test/bfr_dataset_test.csv"
    bfr_tagged_path = "BFR_data_test/bfr_dataset_test_tagged.csv"
    if not os.path.exists(bfr_path):
        bfr_path, bfr_tagged_path = generate_test_data(
            dim=dim_bfr,
            k=10,  # Valeur par défaut pour la génération
            n=n,
            out_path=bfr_path,
            is_cure=False
        )
    
    # Générer les données CURE si elles n'existent pas
    cure_path = "CURE_data_test/cure_dataset_test.csv"
    cure_tagged_path = "CURE_data_test/cure_dataset_test_tagged.csv"
    if not os.path.exists(cure_path):
        cure_path, cure_tagged_path = generate_test_data(
            dim=dim_cure,
            k=10,  # Valeur par défaut pour la génération
            n=n,
            out_path=cure_path,
            is_cure=True
        )
    
    # Définir les grilles de paramètres pour l'optimisation
    bfr_param_grid = {
        'k': k_values,
        'block_size': block_size_values
    }
    
    cure_param_grid = {
        'k': k_values,
        'block_size': block_size_values
    }
    
    # Optimiser BFR sur les données BFR
    print("\n" + "="*80)
    print("Optimisation de BFR sur les données BFR")
    print("="*80)
    best_bfr_on_bfr_params, best_bfr_on_bfr_accuracy, best_bfr_on_bfr_time, bfr_on_bfr_results = optimize_parameters(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        data_path=bfr_path,
        tagged_path=bfr_tagged_path,
        dim=dim_bfr,
        n=n,
        param_grid=bfr_param_grid
    )
    
    # Optimiser CURE sur les données BFR
    print("\n" + "="*80)
    print("Optimisation de CURE sur les données BFR")
    print("="*80)
    best_cure_on_bfr_params, best_cure_on_bfr_accuracy, best_cure_on_bfr_time, cure_on_bfr_results = optimize_parameters(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        data_path=bfr_path,
        tagged_path=bfr_tagged_path,
        dim=dim_bfr,
        n=n,
        param_grid=cure_param_grid
    )
    
    # Optimiser BFR sur les données CURE
    print("\n" + "="*80)
    print("Optimisation de BFR sur les données CURE")
    print("="*80)
    best_bfr_on_cure_params, best_bfr_on_cure_accuracy, best_bfr_on_cure_time, bfr_on_cure_results = optimize_parameters(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        data_path=cure_path,
        tagged_path=cure_tagged_path,
        dim=dim_cure,
        n=n,
        param_grid=bfr_param_grid
    )
    
    # Optimiser CURE sur les données CURE
    print("\n" + "="*80)
    print("Optimisation de CURE sur les données CURE")
    print("="*80)
    best_cure_on_cure_params, best_cure_on_cure_accuracy, best_cure_on_cure_time, cure_on_cure_results = optimize_parameters(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        data_path=cure_path,
        tagged_path=cure_tagged_path,
        dim=dim_cure,
        n=n,
        param_grid=cure_param_grid
    )
    
    # Tracer les graphiques pour visualiser l'impact des paramètres
    for param in ['k', 'block_size']:
        # BFR sur données BFR
        plot_results(bfr_on_bfr_results, param, "BFR_on_BFR", 'accuracy')
        plot_results(bfr_on_bfr_results, param, "BFR_on_BFR", 'time')
        
        # CURE sur données BFR
        plot_results(cure_on_bfr_results, param, "CURE_on_BFR", 'accuracy')
        plot_results(cure_on_bfr_results, param, "CURE_on_BFR", 'time')
        
        # BFR sur données CURE
        plot_results(bfr_on_cure_results, param, "BFR_on_CURE", 'accuracy')
        plot_results(bfr_on_cure_results, param, "BFR_on_CURE", 'time')
        
        # CURE sur données CURE
        plot_results(cure_on_cure_results, param, "CURE_on_CURE", 'accuracy')
        plot_results(cure_on_cure_results, param, "CURE_on_CURE", 'time')
    
    # Afficher le tableau de résultats
    print("\n" + "="*80)
    print("Résumé des résultats optimisés")
    print("="*80)
    
    results = [
        ["BFR sur données BFR", dim_bfr, best_bfr_on_bfr_params['k'], n, best_bfr_on_bfr_params['block_size'], best_bfr_on_bfr_time, best_bfr_on_bfr_accuracy],
        ["CURE sur données BFR", dim_bfr, best_cure_on_bfr_params['k'], n, best_cure_on_bfr_params['block_size'], best_cure_on_bfr_time, best_cure_on_bfr_accuracy],
        ["BFR sur données CURE", dim_cure, best_bfr_on_cure_params['k'], n, best_bfr_on_cure_params['block_size'], best_bfr_on_cure_time, best_bfr_on_cure_accuracy],
        ["CURE sur données CURE", dim_cure, best_cure_on_cure_params['k'], n, best_cure_on_cure_params['block_size'], best_cure_on_cure_time, best_cure_on_cure_accuracy]
    ]
    
    # Afficher le tableau
    print("\nTableau de résultats optimisés:")
    print("-" * 100)
    print(f"{'Test':<20} | {'Dim':<5} | {'K':<5} | {'Points':<10} | {'Bloc':<10} | {'Temps (s)':<10} | {'Précision':<10}")
    print("-" * 100)
    
    for row in results:
        print(f"{row[0]:<20} | {row[1]:<5} | {row[2]:<5} | {row[3]:<10} | {row[4]:<10} | {row[5]:<10.2f} | {row[6]:<10.4f}")
    
    print("-" * 100)
    
    # Sauvegarder les résultats dans un fichier CSV
    with open("secondPart/optimized_results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Dimension", "K", "Points", "Taille de bloc", "Temps (s)", "Précision"])
        writer.writerows(results)
    
    print("\nRésultats optimisés sauvegardés dans 'secondPart/optimized_results_summary.csv'")
    
    # Afficher les conclusions
    print("\nConclusions après optimisation:")
    print(f"1. Sur les données BFR (clusters gaussiens):")
    print(f"   - BFR: Précision = {best_bfr_on_bfr_accuracy:.4f}, Temps = {best_bfr_on_bfr_time:.2f}s, K = {best_bfr_on_bfr_params['k']}, Bloc = {best_bfr_on_bfr_params['block_size']}")
    print(f"   - CURE: Précision = {best_cure_on_bfr_accuracy:.4f}, Temps = {best_cure_on_bfr_time:.2f}s, K = {best_cure_on_bfr_params['k']}, Bloc = {best_cure_on_bfr_params['block_size']}")
    
    print(f"\n2. Sur les données CURE (clusters de formes variées):")
    print(f"   - BFR: Précision = {best_bfr_on_cure_accuracy:.4f}, Temps = {best_bfr_on_cure_time:.2f}s, K = {best_bfr_on_cure_params['k']}, Bloc = {best_bfr_on_cure_params['block_size']}")
    print(f"   - CURE: Précision = {best_cure_on_cure_accuracy:.4f}, Temps = {best_cure_on_cure_time:.2f}s, K = {best_cure_on_cure_params['k']}, Bloc = {best_cure_on_cure_params['block_size']}")
    
    if best_bfr_on_bfr_accuracy > best_cure_on_bfr_accuracy:
        print("\n→ BFR est plus précis sur les données BFR (clusters gaussiens) après optimisation")
    else:
        print("\n→ CURE est plus précis sur les données BFR (clusters gaussiens) après optimisation")
    
    if best_bfr_on_cure_accuracy > best_cure_on_cure_accuracy:
        print("→ BFR est plus précis sur les données CURE (clusters de formes variées) après optimisation")
    else:
        print("→ CURE est plus précis sur les données CURE (clusters de formes variées) après optimisation")
    
    print("\nPour générer des fichiers de 10 Go et exécuter les tests sur ces fichiers avec les paramètres optimisés,")
    print("utilisez les valeurs de k et block_size identifiées comme optimales dans ce script.")

if __name__ == "__main__":
    main() 