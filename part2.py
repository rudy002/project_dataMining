import numpy as np
import csv
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import KMeans
import math
from collections import defaultdict

def generate_large_dataset(dim: int, k: int, n: int, out_path: str, batch_size: int = 1000):
    """
    Génère un grand ensemble de données en écrivant directement dans un fichier CSV,
    sans garder toutes les données en mémoire.
    
    Args:
        dim: nombre de dimensions
        k: nombre de clusters
        n: nombre total de points
        out_path: chemin du fichier de sortie
        batch_size: nombre de points générés par lot
    """
    # Générer les centres des clusters
    np.random.seed(42)
    centers = np.random.uniform(-10, 10, size=(k, dim))
    
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        points_generated = 0
        
        while points_generated < n:
            # Déterminer la taille du lot actuel
            current_batch = min(batch_size, n - points_generated)
            
            # Générer un lot de points
            cluster_indices = np.random.randint(0, k, size=current_batch)
            for i in range(current_batch):
                center = centers[cluster_indices[i]]
                point = np.random.normal(loc=center, scale=1.5, size=dim)
                writer.writerow(point)
            
            points_generated += current_batch
            print(f"Généré {points_generated}/{n} points")

def bfr_cluster(dim: int, k: Optional[int], n: int, block_size: int, 
                in_path: str, out_path: str) -> None:
    """
    Implémente l'algorithme BFR (Bradley-Fayyad-Reina) pour le clustering de grands ensembles.
    
    Args:
        dim: nombre de dimensions
        k: nombre de clusters (None pour détermination automatique)
        n: nombre de points à traiter
        block_size: taille des blocs de données à charger en mémoire
        in_path: chemin du fichier d'entrée
        out_path: chemin du fichier de sortie
    """
    # Phase 1: Déterminer k si non spécifié
    if k is None:
        k = determine_k(in_path, dim, block_size)
    
    # Initialisation des statistiques des clusters
    cluster_stats = {
        i: {
            'n': 0,           # nombre de points
            'sum': np.zeros(dim),    # somme des coordonnées
            'sum_sq': np.zeros(dim)  # somme des carrés
        } for i in range(k)
    }
    
    # Phase 2: Traitement par lots
    with open(in_path, 'r') as infile, open(out_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        buffer = []
        points_processed = 0
        
        for row in reader:
            if points_processed >= n:
                break
                
            point = np.array([float(x) for x in row[:dim]])
            buffer.append(point)
            
            if len(buffer) >= block_size:
                process_block(buffer, cluster_stats, k, writer)
                buffer = []
                points_processed += block_size
                print(f"Traité {points_processed}/{n} points")
        
        # Traiter le dernier bloc s'il existe
        if buffer:
            process_block(buffer, cluster_stats, k, writer)

def process_block(points: List[np.ndarray], stats: Dict, k: int, writer: csv.writer):
    """
    Traite un bloc de points pour BFR.
    """
    # Utiliser k-means pour assigner les points aux clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(points)
    
    # Mettre à jour les statistiques et écrire les résultats
    for point, label in zip(points, labels):
        # Mise à jour des statistiques
        stats[label]['n'] += 1
        stats[label]['sum'] += point
        stats[label]['sum_sq'] += point ** 2
        
        # Écrire le point et son cluster
        writer.writerow(list(point) + [label])

def determine_k(in_path: str, dim: int, block_size: int) -> int:
    """
    Détermine automatiquement le nombre de clusters k.
    """
    # Charger un échantillon de données
    sample = []
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            point = [float(x) for x in row[:dim]]
            sample.append(point)
    
    # Utiliser l'elbow method
    distortions = []
    k_range = range(1, min(11, len(sample)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(sample)
        distortions.append(kmeans.inertia_)
    
    # Trouver le coude
    k_optimal = find_elbow(distortions)
    return k_optimal + 1

def find_elbow(distortions: List[float]) -> int:
    """
    Trouve le point de coude dans la courbe des distortions.
    """
    diffs = np.diff(distortions)
    return np.argmax(diffs) + 1

def cure_cluster(dim: int, k: Optional[int], n: int, block_size: int,
                 in_path: str, out_path: str) -> None:
    """
    Implémente l'algorithme CURE pour le clustering de grands ensembles.
    
    Args:
        dim: nombre de dimensions
        k: nombre de clusters (None pour détermination automatique)
        n: nombre de points à traiter
        block_size: taille des blocs de données
        in_path: chemin du fichier d'entrée
        out_path: chemin du fichier de sortie
    """
    # Phase 1: Échantillonnage et détermination de k si nécessaire
    sample = []
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= block_size:
                break
            point = np.array([float(x) for x in row[:dim]])
            sample.append(point)
    
    if k is None:
        k = determine_k(in_path, dim, block_size)
    
    # Phase 2: CURE clustering sur l'échantillon
    representatives = cure_process(sample, k)
    
    # Phase 3: Assigner les points restants
    with open(in_path, 'r') as infile, open(out_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        points_processed = 0
        for row in reader:
            if points_processed >= n:
                break
                
            point = np.array([float(x) for x in row[:dim]])
            cluster = assign_to_cluster(point, representatives)
            writer.writerow(list(point) + [cluster])
            points_processed += 1
            
            if points_processed % block_size == 0:
                print(f"Traité {points_processed}/{n} points")

def cure_process(points: List[np.ndarray], k: int, num_representatives: int = 10) -> Dict:
    """
    Applique l'algorithme CURE sur un ensemble de points.
    """
    # Initialiser les clusters avec k-means
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(points)
    
    # Trouver les représentants pour chaque cluster
    representatives = defaultdict(list)
    for i in range(k):
        cluster_points = [p for j, p in enumerate(points) if labels[j] == i]
        if cluster_points:
            # Sélectionner les points les plus dispersés comme représentants
            reps = select_representatives(cluster_points, num_representatives)
            representatives[i] = reps
    
    return representatives

def select_representatives(points: List[np.ndarray], num_reps: int) -> List[np.ndarray]:
    """
    Sélectionne les points représentatifs pour CURE.
    """
    if len(points) <= num_reps:
        return points
    
    # Trouver les points les plus dispersés
    selected = [points[0]]
    while len(selected) < num_reps:
        max_dist = -1
        max_point = None
        
        for point in points:
            min_dist = min(np.linalg.norm(point - rep) for rep in selected)
            if min_dist > max_dist:
                max_dist = min_dist
                max_point = point
        
        if max_point is not None:
            selected.append(max_point)
    
    return selected

def assign_to_cluster(point: np.ndarray, representatives: Dict) -> int:
    """
    Assigne un point au cluster le plus proche basé sur les représentants.
    """
    min_dist = float('inf')
    best_cluster = 0
    
    for cluster_id, reps in representatives.items():
        for rep in reps:
            dist = np.linalg.norm(point - rep)
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster_id
    
    return best_cluster

if __name__ == "__main__":
    # Test de génération de grand dataset
    print("Génération d'un grand dataset...")
    generate_large_dataset(dim=6, k=4, n=10000, out_path="large_dataset.csv")
    
    # Test de BFR
    print("\nTest de BFR clustering...")
    bfr_cluster(dim=6, k=4, n=10000, block_size=1000,
                in_path="large_dataset.csv", out_path="bfr_results.csv")
    
    # Test de CURE
    print("\nTest de CURE clustering...")
    cure_cluster(dim=6, k=4, n=10000, block_size=1000,
                in_path="large_dataset.csv", out_path="cure_results.csv")
