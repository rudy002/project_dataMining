import os
import sys
import time
import numpy as np
import csv

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstPart.part1 import generate_data, calculate_accuracy
from secondPart.part2 import bfr_cluster, cure_cluster

def generate_test_data(dim, k, n, out_path, is_cure=False):
    """
    Génère des données de test pour les algorithmes BFR et CURE.
    """
    print(f"Génération de données {'CURE' if is_cure else 'BFR'} (dim={dim}, k={k}, n={n})...")
    
    if is_cure:
        # Pour CURE, utiliser une fonction personnalisée pour générer des clusters de formes variées
        generate_data(dim=dim, k=k, n=n, out_path=out_path, points_gen=cure_points_generator)
    else:
        # Pour BFR, utiliser des clusters gaussiens bien séparés
        extras = {
            'std_dev': 0.5,
            'max_coord': 100.0,
            'min_points_per_cluster': n // (2 * k)
        }
        generate_data(dim=dim, k=k, n=n, out_path=out_path, extras=extras)
    
    # Vérifier la taille du fichier
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Taille du fichier: {file_size_mb:.2f} Mo")
    
    return out_path, out_path.replace('.csv', '_tagged.csv')

def cure_points_generator(dim, k, n, extras):
    """
    Fonction personnalisée pour générer des points pour CURE avec des formes de clusters variées.
    """
    points = []
    
    # Distribuer les points parmi les clusters
    points_per_cluster = [n // k] * k
    remaining_points = n - sum(points_per_cluster)
    
    for i in range(remaining_points):
        points_per_cluster[i % k] += 1
    
    # Générer des clusters de formes variées
    for cluster_id in range(k):
        # Choisir aléatoirement un type de cluster
        cluster_type = np.random.choice(['gaussian', 'elongated', 'curved'])
        
        # Centre du cluster
        center = np.random.uniform(0, 100.0, dim)
        
        # Nombre de points pour ce cluster
        num_points = points_per_cluster[cluster_id]
        
        for _ in range(num_points):
            point = np.zeros(dim)
            
            if cluster_type == 'gaussian':
                # Cluster gaussien standard
                std_dev = np.random.uniform(1.0, 5.0)
                point = center + np.random.normal(0, std_dev, dim)
            
            elif cluster_type == 'elongated':
                # Cluster allongé dans une direction aléatoire
                direction = np.random.normal(0, 1, dim)
                direction = direction / np.linalg.norm(direction)
                
                # Position le long de la direction principale
                t = np.random.normal(0, 10.0)
                
                # Écart par rapport à la direction principale
                noise = np.random.normal(0, 1.0, dim)
                
                point = center + t * direction + noise
            
            elif cluster_type == 'curved':
                # Cluster en forme de courbe (pour dim >= 2)
                if dim >= 2:
                    # Paramètre pour la courbe
                    t = np.random.uniform(-np.pi, np.pi)
                    
                    # Les deux premières dimensions forment une courbe
                    point[0] = center[0] + 10 * np.cos(t)
                    point[1] = center[1] + 10 * np.sin(t)
                    
                    # Autres dimensions avec bruit gaussien
                    if dim > 2:
                        point[2:] = center[2:] + np.random.normal(0, 1.0, dim-2)
                else:
                    # Fallback pour dim=1
                    point = center + np.random.normal(0, 3.0, dim)
            
            points.append((tuple(point), cluster_id))
    
    return points

def test_algorithm(algorithm_name, algorithm_func, dim, k, n, block_size, in_path, out_path, tagged_path):
    """
    Teste un algorithme de clustering sur un jeu de données et mesure ses performances.
    """
    print(f"\nTest de l'algorithme {algorithm_name}:")
    print(f"Fichier d'entrée: {in_path}")
    print(f"Dimension: {dim}, Clusters: {k}, Points: {n}, Taille de bloc: {block_size}")
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    
    # Exécuter l'algorithme
    clusters = algorithm_func(
        dim=dim,
        k=k,
        n=n,
        block_size=block_size,
        in_path=in_path,
        out_path=out_path
    )
    
    execution_time = time.time() - start_time
    print(f"Temps d'exécution: {execution_time:.2f} secondes")
    
    # Afficher les résultats
    print(f"Nombre de clusters créés: {len(clusters)}")
    
    # Calculer la précision
    formatted_clusters = []
    for cluster in clusters:
        if 'points' in cluster:
            # Convertir les points en tuples pour qu'ils soient hashables
            formatted_points = [tuple(p) if hasattr(p, '__iter__') else p for p in cluster['points']]
            formatted_clusters.append(formatted_points)
        else:
            # Pour BFR, les points sont stockés différemment
            formatted_points = [tuple(p) if hasattr(p, '__iter__') else p for p in cluster.get('points', [])]
            formatted_clusters.append(formatted_points)
    
    # Créer une version personnalisée de calculate_accuracy qui fonctionne avec nos données
    accuracy = calculate_cluster_accuracy(formatted_clusters, tagged_path, dim)
    print(f"Précision: {accuracy:.4f}")
    
    return clusters, execution_time, accuracy

def calculate_cluster_accuracy(clusters, tagged_file_path, dim):
    """
    Calcule la précision du clustering en comparant avec les étiquettes réelles.
    Version adaptée pour fonctionner avec les formats de données de BFR et CURE.
    """
    # Charger les points étiquetés
    tagged_points = []
    with open(tagged_file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) > dim:  # S'assurer qu'il y a suffisamment de valeurs
                point = tuple(float(values[i]) for i in range(dim))
                cluster_id = int(values[dim])
                tagged_points.append((point, cluster_id))
    
    # Créer un dictionnaire pour mapper les points à leurs clusters réels
    true_clusters = {}
    for point, cluster_id in tagged_points:
        if cluster_id not in true_clusters:
            true_clusters[cluster_id] = []
        true_clusters[cluster_id].append(point)
    
    # Créer un dictionnaire pour mapper les points à leurs clusters prédits
    pred_clusters = {}
    for i, cluster in enumerate(clusters):
        for point in cluster:
            pred_clusters[point] = i
    
    # Calculer la matrice de confusion entre les clusters réels et prédits
    confusion_matrix = {}
    for true_id, true_cluster in true_clusters.items():
        confusion_matrix[true_id] = {}
        for pred_id in range(len(clusters)):
            confusion_matrix[true_id][pred_id] = 0
        
        for point in true_cluster:
            if point in pred_clusters:
                pred_id = pred_clusters[point]
                confusion_matrix[true_id][pred_id] += 1
    
    # Trouver la meilleure correspondance entre les clusters réels et prédits
    best_matches = {}
    remaining_pred = list(range(len(clusters)))
    
    for true_id in sorted(true_clusters.keys(), key=lambda k: len(true_clusters[k]), reverse=True):
        best_pred = -1
        best_count = -1
        
        for pred_id in remaining_pred:
            count = confusion_matrix[true_id][pred_id]
            if count > best_count:
                best_count = count
                best_pred = pred_id
        
        if best_pred != -1:
            best_matches[true_id] = best_pred
            remaining_pred.remove(best_pred)
    
    # Calculer la précision
    correct = 0
    total = 0
    
    for point, true_id in tagged_points:
        if point in pred_clusters:
            pred_id = pred_clusters[point]
            if true_id in best_matches and best_matches[true_id] == pred_id:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    # Créer les répertoires
    os.makedirs("BFR_data_test", exist_ok=True)
    os.makedirs("CURE_data_test", exist_ok=True)
    
    # Paramètres
    n = 10000        # Nombre de points (petit pour les tests)
    dim_bfr = 5      # Dimension pour BFR
    dim_cure = 6     # Dimension pour CURE
    k = 10           # Nombre de clusters
    block_size = 1000  # Taille du bloc pour les algorithmes
    
    # Générer les données BFR
    bfr_path, bfr_tagged_path = generate_test_data(
        dim=dim_bfr,
        k=k,
        n=n,
        out_path="BFR_data_test/bfr_dataset_test.csv",
        is_cure=False
    )
    
    # Générer les données CURE
    cure_path, cure_tagged_path = generate_test_data(
        dim=dim_cure,
        k=k,
        n=n,
        out_path="CURE_data_test/cure_dataset_test.csv",
        is_cure=True
    )
    
    # Tester BFR sur les données BFR
    print("\n" + "="*80)
    print("Test de BFR sur les données BFR")
    print("="*80)
    bfr_on_bfr_clusters, bfr_on_bfr_time, bfr_on_bfr_accuracy = test_algorithm(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        dim=dim_bfr,
        k=k,
        n=n,
        block_size=block_size,
        in_path=bfr_path,
        out_path="BFR_data_test/bfr_results_bfr.csv",
        tagged_path=bfr_tagged_path
    )
    
    # Tester CURE sur les données BFR
    print("\n" + "="*80)
    print("Test de CURE sur les données BFR")
    print("="*80)
    cure_on_bfr_clusters, cure_on_bfr_time, cure_on_bfr_accuracy = test_algorithm(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        dim=dim_bfr,
        k=k,
        n=n,
        block_size=block_size,
        in_path=bfr_path,
        out_path="BFR_data_test/cure_results_bfr.csv",
        tagged_path=bfr_tagged_path
    )
    
    # Tester BFR sur les données CURE
    print("\n" + "="*80)
    print("Test de BFR sur les données CURE")
    print("="*80)
    bfr_on_cure_clusters, bfr_on_cure_time, bfr_on_cure_accuracy = test_algorithm(
        algorithm_name="BFR",
        algorithm_func=bfr_cluster,
        dim=dim_cure,
        k=k,
        n=n,
        block_size=block_size,
        in_path=cure_path,
        out_path="CURE_data_test/bfr_results_cure.csv",
        tagged_path=cure_tagged_path
    )
    
    # Tester CURE sur les données CURE
    print("\n" + "="*80)
    print("Test de CURE sur les données CURE")
    print("="*80)
    cure_on_cure_clusters, cure_on_cure_time, cure_on_cure_accuracy = test_algorithm(
        algorithm_name="CURE",
        algorithm_func=cure_cluster,
        dim=dim_cure,
        k=k,
        n=n,
        block_size=block_size,
        in_path=cure_path,
        out_path="CURE_data_test/cure_results_cure.csv",
        tagged_path=cure_tagged_path
    )
    
    # Afficher le tableau de résultats
    print("\n" + "="*80)
    print("Résumé des résultats")
    print("="*80)
    
    results = [
        ["BFR sur données BFR", dim_bfr, k, n, block_size, bfr_on_bfr_time, bfr_on_bfr_accuracy],
        ["CURE sur données BFR", dim_bfr, k, n, block_size, cure_on_bfr_time, cure_on_bfr_accuracy],
        ["BFR sur données CURE", dim_cure, k, n, block_size, bfr_on_cure_time, bfr_on_cure_accuracy],
        ["CURE sur données CURE", dim_cure, k, n, block_size, cure_on_cure_time, cure_on_cure_accuracy]
    ]
    
    # Afficher le tableau
    print("\nTableau de résultats:")
    print("-" * 100)
    print(f"{'Test':<20} | {'Dim':<5} | {'K':<5} | {'Points':<10} | {'Bloc':<10} | {'Temps (s)':<10} | {'Précision':<10}")
    print("-" * 100)
    
    for row in results:
        print(f"{row[0]:<20} | {row[1]:<5} | {row[2]:<5} | {row[3]:<10} | {row[4]:<10} | {row[5]:<10.2f} | {row[6]:<10.4f}")
    
    print("-" * 100)
    
    # Sauvegarder les résultats dans un fichier CSV
    with open("secondPart/results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Dimension", "K", "Points", "Taille de bloc", "Temps (s)", "Précision"])
        writer.writerows(results)
    
    print("\nRésultats sauvegardés dans 'secondPart/results_summary.csv'")
    
    # Afficher les conclusions
    print("\nConclusions:")
    print(f"1. Sur les données BFR (clusters gaussiens):")
    print(f"   - BFR: Précision = {bfr_on_bfr_accuracy:.4f}, Temps = {bfr_on_bfr_time:.2f}s")
    print(f"   - CURE: Précision = {cure_on_bfr_accuracy:.4f}, Temps = {cure_on_bfr_time:.2f}s")
    
    print(f"\n2. Sur les données CURE (clusters de formes variées):")
    print(f"   - BFR: Précision = {bfr_on_cure_accuracy:.4f}, Temps = {bfr_on_cure_time:.2f}s")
    print(f"   - CURE: Précision = {cure_on_cure_accuracy:.4f}, Temps = {cure_on_cure_time:.2f}s")
    
    if bfr_on_bfr_accuracy > cure_on_bfr_accuracy:
        print("\n→ BFR est plus précis sur les données BFR (clusters gaussiens)")
    else:
        print("\n→ CURE est plus précis sur les données BFR (clusters gaussiens)")
    
    if bfr_on_cure_accuracy > cure_on_cure_accuracy:
        print("→ BFR est plus précis sur les données CURE (clusters de formes variées)")
    else:
        print("→ CURE est plus précis sur les données CURE (clusters de formes variées)")
    
    print("\nPour générer des fichiers de 10 Go et exécuter les tests sur ces fichiers,")
    print("modifiez la valeur de n dans le script (n = 235000000 pour BFR, n = 195000000 pour CURE).")

if __name__ == "__main__":
    main() 