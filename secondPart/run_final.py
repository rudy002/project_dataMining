import os
import sys
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstPart.part1 import generate_data
from secondPart.part2 import bfr_cluster, cure_cluster
from secondPart.run_part2 import generate_test_data, cure_points_generator, calculate_cluster_accuracy

def run_algorithm_with_optimal_params(algorithm_name, algorithm_func, data_path, tagged_path, dim, n, optimal_params):
    """
    Exécute un algorithme avec les paramètres optimaux et mesure ses performances.
    """
    print(f"\nExécution de {algorithm_name} avec les paramètres optimaux:")
    print(f"Paramètres: {optimal_params}")
    print(f"Fichier d'entrée: {data_path}")
    
    # Préparer les arguments
    kwargs = {
        'dim': dim,
        'n': n,
        'in_path': data_path,
        'out_path': f"{os.path.dirname(data_path)}/{algorithm_name.lower()}_results_optimal.csv"
    }
    kwargs.update(optimal_params)
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    
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
    
    print(f"Nombre de clusters créés: {len(clusters)}")
    print(f"Précision: {accuracy:.4f}")
    print(f"Temps d'exécution: {execution_time:.2f} secondes")
    
    return clusters, execution_time, accuracy

def plot_comparison(results, title, filename):
    """
    Crée un graphique de comparaison des algorithmes.
    """
    algorithms = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    times = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique de précision
    ax1.bar(algorithms, accuracies, color=['blue', 'orange'])
    ax1.set_ylabel('Précision')
    ax1.set_title('Comparaison de la précision')
    ax1.set_ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Graphique de temps
    ax2.bar(algorithms, times, color=['blue', 'orange'])
    ax2.set_ylabel('Temps (secondes)')
    ax2.set_title('Comparaison du temps d\'exécution')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(times):
        ax2.text(i, v + 0.02, f"{v:.2f}s", ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(f"secondPart/{filename}.png")
    plt.close()

def main():
    # Créer les répertoires
    os.makedirs("BFR_data_test", exist_ok=True)
    os.makedirs("CURE_data_test", exist_ok=True)
    
    # Paramètres
    n = 10000        # Nombre de points (petit pour les tests)
    dim_bfr = 5      # Dimension pour BFR
    dim_cure = 6     # Dimension pour CURE
    
    # Paramètres optimaux trouvés précédemment
    bfr_on_bfr_optimal = {'k': 20, 'block_size': 500}
    cure_on_bfr_optimal = {'k': 15, 'block_size': 1000}
    bfr_on_cure_optimal = {'k': 15, 'block_size': 1000}
    cure_on_cure_optimal = {'k': 15, 'block_size': 500}
    
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
    
    # Exécuter BFR sur les données BFR avec les paramètres optimaux
    print("\n" + "="*80)
    print("Exécution de BFR sur les données BFR avec les paramètres optimaux")
    print("="*80)
    _, bfr_on_bfr_time, bfr_on_bfr_accuracy = run_algorithm_with_optimal_params(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        data_path=bfr_path,
        tagged_path=bfr_tagged_path,
        dim=dim_bfr,
        n=n,
        optimal_params=bfr_on_bfr_optimal
    )
    
    # Exécuter CURE sur les données BFR avec les paramètres optimaux
    print("\n" + "="*80)
    print("Exécution de CURE sur les données BFR avec les paramètres optimaux")
    print("="*80)
    _, cure_on_bfr_time, cure_on_bfr_accuracy = run_algorithm_with_optimal_params(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        data_path=bfr_path,
        tagged_path=bfr_tagged_path,
        dim=dim_bfr,
        n=n,
        optimal_params=cure_on_bfr_optimal
    )
    
    # Exécuter BFR sur les données CURE avec les paramètres optimaux
    print("\n" + "="*80)
    print("Exécution de BFR sur les données CURE avec les paramètres optimaux")
    print("="*80)
    _, bfr_on_cure_time, bfr_on_cure_accuracy = run_algorithm_with_optimal_params(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        data_path=cure_path,
        tagged_path=cure_tagged_path,
        dim=dim_cure,
        n=n,
        optimal_params=bfr_on_cure_optimal
    )
    
    # Exécuter CURE sur les données CURE avec les paramètres optimaux
    print("\n" + "="*80)
    print("Exécution de CURE sur les données CURE avec les paramètres optimaux")
    print("="*80)
    _, cure_on_cure_time, cure_on_cure_accuracy = run_algorithm_with_optimal_params(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        data_path=cure_path,
        tagged_path=cure_tagged_path,
        dim=dim_cure,
        n=n,
        optimal_params=cure_on_cure_optimal
    )
    
    # Créer des graphiques de comparaison
    bfr_data_results = [
        ("BFR", bfr_on_bfr_accuracy, bfr_on_bfr_time),
        ("CURE", cure_on_bfr_accuracy, cure_on_bfr_time)
    ]
    
    cure_data_results = [
        ("BFR", bfr_on_cure_accuracy, bfr_on_cure_time),
        ("CURE", cure_on_cure_accuracy, cure_on_cure_time)
    ]
    
    plot_comparison(bfr_data_results, "Comparaison sur données BFR (clusters gaussiens)", "comparison_bfr_data")
    plot_comparison(cure_data_results, "Comparaison sur données CURE (clusters de formes variées)", "comparison_cure_data")
    
    # Afficher le tableau de résultats
    print("\n" + "="*80)
    print("Résumé des résultats finaux")
    print("="*80)
    
    results = [
        ["BFR sur données BFR", dim_bfr, bfr_on_bfr_optimal['k'], n, bfr_on_bfr_optimal['block_size'], bfr_on_bfr_time, bfr_on_bfr_accuracy],
        ["CURE sur données BFR", dim_bfr, cure_on_bfr_optimal['k'], n, cure_on_bfr_optimal['block_size'], cure_on_bfr_time, cure_on_bfr_accuracy],
        ["BFR sur données CURE", dim_cure, bfr_on_cure_optimal['k'], n, bfr_on_cure_optimal['block_size'], bfr_on_cure_time, bfr_on_cure_accuracy],
        ["CURE sur données CURE", dim_cure, cure_on_cure_optimal['k'], n, cure_on_cure_optimal['block_size'], cure_on_cure_time, cure_on_cure_accuracy]
    ]
    
    # Afficher le tableau
    print("\nTableau de résultats finaux:")
    print("-" * 100)
    print(f"{'Test':<20} | {'Dim':<5} | {'K':<5} | {'Points':<10} | {'Bloc':<10} | {'Temps (s)':<10} | {'Précision':<10}")
    print("-" * 100)
    
    for row in results:
        print(f"{row[0]:<20} | {row[1]:<5} | {row[2]:<5} | {row[3]:<10} | {row[4]:<10} | {row[5]:<10.2f} | {row[6]:<10.4f}")
    
    print("-" * 100)
    
    # Sauvegarder les résultats dans un fichier CSV
    with open("secondPart/final_results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Dimension", "K", "Points", "Taille de bloc", "Temps (s)", "Précision"])
        writer.writerows(results)
    
    print("\nRésultats finaux sauvegardés dans 'secondPart/final_results_summary.csv'")
    
    # Afficher les conclusions
    print("\nConclusions finales:")
    print(f"1. Sur les données BFR (clusters gaussiens):")
    print(f"   - BFR: Précision = {bfr_on_bfr_accuracy:.4f}, Temps = {bfr_on_bfr_time:.2f}s")
    print(f"   - CURE: Précision = {cure_on_bfr_accuracy:.4f}, Temps = {cure_on_bfr_time:.2f}s")
    
    print(f"\n2. Sur les données CURE (clusters de formes variées):")
    print(f"   - BFR: Précision = {bfr_on_cure_accuracy:.4f}, Temps = {bfr_on_cure_time:.2f}s")
    print(f"   - CURE: Précision = {cure_on_cure_accuracy:.4f}, Temps = {cure_on_cure_time:.2f}s")
    
    if bfr_on_bfr_accuracy > cure_on_bfr_accuracy:
        print("\n→ BFR est plus précis sur les données BFR (clusters gaussiens)")
        bfr_advantage = (bfr_on_bfr_accuracy - cure_on_bfr_accuracy) / cure_on_bfr_accuracy * 100
        print(f"   Avantage de {bfr_advantage:.2f}% en précision")
    else:
        print("\n→ CURE est plus précis sur les données BFR (clusters gaussiens)")
        cure_advantage = (cure_on_bfr_accuracy - bfr_on_bfr_accuracy) / bfr_on_bfr_accuracy * 100
        print(f"   Avantage de {cure_advantage:.2f}% en précision")
    
    if bfr_on_cure_accuracy > cure_on_cure_accuracy:
        print("→ BFR est plus précis sur les données CURE (clusters de formes variées)")
        bfr_advantage = (bfr_on_cure_accuracy - cure_on_cure_accuracy) / cure_on_cure_accuracy * 100
        print(f"   Avantage de {bfr_advantage:.2f}% en précision")
    else:
        print("→ CURE est plus précis sur les données CURE (clusters de formes variées)")
        cure_advantage = (cure_on_cure_accuracy - bfr_on_cure_accuracy) / bfr_on_cure_accuracy * 100
        print(f"   Avantage de {cure_advantage:.2f}% en précision")
    
    print("\nPour générer des fichiers de 10 Go et exécuter les tests sur ces fichiers avec les paramètres optimisés,")
    print("utilisez les valeurs de k et block_size identifiées comme optimales dans ce script.")
    print("Paramètres recommandés pour les grands jeux de données:")
    print(f"- BFR: k = {bfr_on_bfr_optimal['k']}, block_size = {bfr_on_bfr_optimal['block_size']}")
    print(f"- CURE: k = {cure_on_cure_optimal['k']}, block_size = {cure_on_cure_optimal['block_size']}")

if __name__ == "__main__":
    main() 