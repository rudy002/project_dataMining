import os
import sys
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstPart.part1 import generate_data, load_points
from secondPart.part2 import bfr_cluster, cure_cluster
from secondPart.run_part2 import calculate_cluster_accuracy

def run_algorithm_with_k(algorithm_name, algorithm_func, data_path, tagged_path, dim, n, k_value, block_size=1000):
    """
    Exécute un algorithme avec une valeur de k spécifique et mesure sa précision.
    """
    print(f"\nExécution de {algorithm_name} avec k={k_value}:")
    
    # Préparer les arguments
    kwargs = {
        'dim': dim,
        'k': k_value,
        'n': n,
        'block_size': block_size,
        'in_path': data_path,
        'out_path': f"secondPart/{algorithm_name.lower()}_k{k_value}_results.csv"
    }
    
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
    
    return accuracy, execution_time

def plot_k_comparison(k_values, bfr_accuracies, cure_accuracies, title, filename):
    """
    Crée un graphique comparant les précisions des algorithmes pour différentes valeurs de k.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, bfr_accuracies, 'o-', label='BFR', color='blue')
    plt.plot(k_values, cure_accuracies, 'o-', label='CURE', color='orange')
    
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Précision')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ajouter les valeurs sur les points
    for i, (bfr_acc, cure_acc) in enumerate(zip(bfr_accuracies, cure_accuracies)):
        plt.text(k_values[i], bfr_acc + 0.01, f"{bfr_acc:.4f}", ha='center', va='bottom')
        plt.text(k_values[i], cure_acc - 0.02, f"{cure_acc:.4f}", ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig(f"secondPart/{filename}.png")
    plt.close()

def main():
    # Vérifier si le fichier d'exemple existe, sinon le générer
    example_data_path = "firstPart/example_data.csv"
    example_tagged_path = "firstPart/example_data_tagged.csv"
    
    dim = 3  # Dimension des données d'exemple
    n = 1000  # Nombre de points
    k_original = 8  # Valeur de k utilisée pour générer les données
    
    if not os.path.exists(example_data_path):
        print(f"Génération des données d'exemple avec k={k_original}...")
        generate_data(dim=dim, k=k_original, n=n, out_path=example_data_path)
    
    # Valeurs de k à tester
    k_values = list(range(2, 9))  # De 2 à 8
    
    # Stocker les résultats
    bfr_accuracies = []
    cure_accuracies = []
    
    # Exécuter BFR et CURE pour chaque valeur de k
    for k in k_values:
        print("\n" + "="*80)
        print(f"Test avec k = {k}")
        print("="*80)
        
        # Exécuter BFR
        bfr_accuracy, _ = run_algorithm_with_k(
            algorithm_name="BFR",
            algorithm_func=bfr_cluster,
            data_path=example_data_path,
            tagged_path=example_tagged_path,
            dim=dim,
            n=n,
            k_value=k
        )
        bfr_accuracies.append(bfr_accuracy)
        
        # Exécuter CURE
        cure_accuracy, _ = run_algorithm_with_k(
            algorithm_name="CURE",
            algorithm_func=cure_cluster,
            data_path=example_data_path,
            tagged_path=example_tagged_path,
            dim=dim,
            n=n,
            k_value=k
        )
        cure_accuracies.append(cure_accuracy)
    
    # Créer un graphique de comparaison
    plot_k_comparison(
        k_values, 
        bfr_accuracies, 
        cure_accuracies, 
        "Comparaison de la précision de BFR et CURE pour différentes valeurs de k",
        "k_comparison"
    )
    
    # Afficher le tableau de résultats
    print("\n" + "="*80)
    print("Résumé des résultats")
    print("="*80)
    
    print("\nTableau de résultats:")
    print("-" * 60)
    print(f"{'k':<5} | {'BFR Précision':<15} | {'CURE Précision':<15}")
    print("-" * 60)
    
    for i, k in enumerate(k_values):
        print(f"{k:<5} | {bfr_accuracies[i]:<15.4f} | {cure_accuracies[i]:<15.4f}")
    
    print("-" * 60)
    
    # Sauvegarder les résultats dans un fichier CSV
    with open("secondPart/k_comparison_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "BFR Précision", "CURE Précision"])
        for i, k in enumerate(k_values):
            writer.writerow([k, bfr_accuracies[i], cure_accuracies[i]])
    
    print("\nRésultats sauvegardés dans 'secondPart/k_comparison_results.csv'")
    
    # Trouver la meilleure valeur de k pour chaque algorithme
    best_k_bfr = k_values[np.argmax(bfr_accuracies)]
    best_k_cure = k_values[np.argmax(cure_accuracies)]
    
    print("\nMeilleures valeurs de k:")
    print(f"- BFR: k = {best_k_bfr} (Précision: {max(bfr_accuracies):.4f})")
    print(f"- CURE: k = {best_k_cure} (Précision: {max(cure_accuracies):.4f})")
    
    # Comparer avec la valeur de k originale
    print(f"\nValeur de k utilisée pour générer les données: {k_original}")
    print(f"- BFR avec k={k_original}: Précision = {bfr_accuracies[k_original-2]:.4f}")
    print(f"- CURE avec k={k_original}: Précision = {cure_accuracies[k_original-2]:.4f}")

if __name__ == "__main__":
    main() 