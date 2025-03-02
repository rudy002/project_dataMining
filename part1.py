import numpy as np
import csv
from typing import List, Tuple, Optional, Callable, Dict
import math

def generate_data(dim: int, k: int, n: int, out_path: str, 
                 points_gen: Optional[Callable] = None, 
                 extras: Dict = {}) -> List[List[Tuple]]:
    """
    Génère un ensemble de données en `dim` dimensions,
    réparti en `k` clusters et sauvegarde dans un fichier CSV.
    
    Args:
        dim: nombre de dimensions (>0)
        k: nombre de clusters (>0)
        n: nombre total de points (>=k)
        out_path: chemin du fichier de sortie
        points_gen: fonction optionnelle pour générer les points
        extras: dictionnaire de paramètres supplémentaires
            - seed: graine pour la génération aléatoire (défaut: 42)
            - scale: échelle pour la distribution normale (défaut: 1.5)
            - shuffle: mélanger les points (défaut: True)
            - add_labels: ajouter les étiquettes des clusters (défaut: False)
    
    Returns:
        Liste des clusters générés
    """
    # Validation des paramètres
    if dim <= 0 or k <= 0 or n <= 0:
        raise ValueError("dim, k et n doivent être positifs")
    if n < k:
        raise ValueError("n doit être supérieur ou égal à k")
    
    np.random.seed(extras.get('seed', 42))
    clusters = []
    points_per_cluster = n // k
    
    def default_points_gen(center, n_points, dim, scale=1.5):
        """Génère des points selon une distribution normale autour d'un centre."""
        return np.random.normal(loc=center, scale=scale, size=(n_points, dim))
    
    generator = points_gen if points_gen is not None else default_points_gen
    
    for i in range(k):
        center = np.random.uniform(-10, 10, dim)
        scale = extras.get('scale', 1.5)
        points = generator(center, points_per_cluster, dim, scale)
        clusters.append([tuple(p) for p in points])
    
    # Préparer tous les points pour le fichier CSV
    all_points = []
    for i, cluster in enumerate(clusters):
        all_points.extend(cluster)
    
    if extras.get('shuffle', True):
        np.random.shuffle(all_points)
    
    # Ajouter les labels si demandé
    if extras.get('add_labels', False):
        labeled_points = []
        for i, cluster in enumerate(clusters):
            labeled_points.extend([p + (i,) for p in cluster])
        all_points = labeled_points
    
    # Sauvegarder dans le fichier CSV
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for point in all_points:
            writer.writerow(point)
    
    print(f"Fichier {out_path} généré avec {n} points en {dim} dimensions.")
    return clusters

def load_points(in_path: str, dim: int, n: int = -1, 
               points: List[Tuple] = None) -> List[Tuple]:
    """
    Charge des points depuis un fichier CSV.
    
    Args:
        in_path: chemin du fichier d'entrée
        dim: nombre de dimensions à charger
        n: nombre de points à charger (-1 pour tous)
        points: liste existante à laquelle ajouter les points
    
    Returns:
        Liste de tuples représentant les points
    """
    if points is None:
        points = []
    
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if n != -1 and i >= n:
                break
            try:
                point = tuple(map(float, row[:dim]))
                points.append(point)
            except (ValueError, IndexError) as e:
                print(f"Erreur lors de la lecture de la ligne {i+1}: {e}")
                continue
    
    print(f"Chargé {len(points)} points depuis {in_path}.")
    return points

def h_clustering(dim: int, k: Optional[int], points: List[Tuple],
                dist: Optional[Callable] = None,
                clusts: List[List[Tuple]] = None) -> List[List[Tuple]]:
    """
    Effectue un clustering hiérarchique bottom-up.
    
    Args:
        dim: nombre de dimensions
        k: nombre de clusters souhaité (None pour clustering complet)
        points: liste des points à clusteriser
        dist: fonction de distance (None pour distance euclidienne)
        clusts: clusters initiaux (None pour commencer avec des singletons)
    
    Returns:
        Liste des clusters finaux
    """
    def euclidean_distance(p1: Tuple, p2: Tuple) -> float:
        """Calcule la distance euclidienne entre deux points."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))
    
    def find_closest_clusters(clusters: List[List[Tuple]]) -> Tuple[int, int, float]:
        """Trouve les deux clusters les plus proches."""
        min_dist = float('inf')
        min_i = min_j = 0
        distance_func = dist if dist is not None else euclidean_distance
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Distance minimale entre tous les points des deux clusters
                cluster_dist = float('inf')
                for p1 in clusters[i]:
                    for p2 in clusters[j]:
                        d = distance_func(p1, p2)
                        cluster_dist = min(cluster_dist, d)
                
                if cluster_dist < min_dist:
                    min_dist = cluster_dist
                    min_i, min_j = i, j
        
        return min_i, min_j, min_dist
    
    # Initialiser les clusters si non fournis
    if clusts is None:
        clusts = [[p] for p in points]
    
    # Si k est None, continuer jusqu'à avoir un seul cluster
    target_k = k if k is not None else 1
    
    while len(clusts) > target_k:
        # Trouver et fusionner les deux clusters les plus proches
        i, j, min_dist = find_closest_clusters(clusts)
        new_cluster = clusts[i] + clusts[j]
        clusts = [c for idx, c in enumerate(clusts) if idx not in (i, j)]
        clusts.append(new_cluster)
    
    return clusts

def save_points(clusts: List[List[Tuple]], out_path: str, 
                out_path_tagged: str) -> None:
    """
    Sauvegarde les points dans deux fichiers CSV.
    
    Args:
        clusts: liste des clusters (liste de listes de points)
        out_path: chemin pour sauvegarder les points sans étiquettes
        out_path_tagged: chemin pour sauvegarder les points avec étiquettes
    """
    # Préparer les points avec et sans étiquettes
    points = []
    points_tagged = []
    
    for cluster_idx, cluster in enumerate(clusts):
        for point in cluster:
            points.append(point)
            points_tagged.append(point + (cluster_idx,))
    
    # Mélanger les points tout en gardant la correspondance
    indices = list(range(len(points)))
    np.random.shuffle(indices)
    
    points = [points[i] for i in indices]
    points_tagged = [points_tagged[i] for i in indices]
    
    # Sauvegarder les points sans étiquettes
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for point in points:
            writer.writerow(point)
    
    # Sauvegarder les points avec étiquettes
    with open(out_path_tagged, 'w', newline='') as file:
        writer = csv.writer(file)
        for point in points_tagged:
            writer.writerow(point)
    
    print(f"Points sauvegardés dans {out_path} et {out_path_tagged}")

if __name__ == "__main__":
    print("Début des tests avec les fichiers fournis...")
    
    # Test avec données 3D
    print("\n=== Test avec données 3D ===")
    points_3d = load_points("ClusteringFiles/points_3d_01_example01.csv", dim=3)
    clusters_3d = h_clustering(dim=3, k=4, points=points_3d)  # k=4 pour ce jeu de données
    save_points(
        clusters_3d,
        "resultats_3d.csv",
        "resultats_3d_tagged.csv"
    )
    print(f"Nombre de clusters 3D: {len(clusters_3d)}")
    for i, cluster in enumerate(clusters_3d):
        print(f"Cluster {i}: {len(cluster)} points")
    
    # Test avec données 6D
    print("\n=== Test avec données 6D ===")
    points_6d = load_points("ClusteringFiles/points_4clusts_6d_example03.CSV", dim=6)
    clusters_6d = h_clustering(dim=6, k=4, points=points_6d)  # k=4 comme indiqué dans le nom du fichier
    save_points(
        clusters_6d,
        "resultats_6d.csv",
        "resultats_6d_tagged.csv"
    )
    print(f"Nombre de clusters 6D: {len(clusters_6d)}")
    for i, cluster in enumerate(clusters_6d):
        print(f"Cluster {i}: {len(cluster)} points")
    
    # Test avec données 4D
    print("\n=== Test avec données 4D ===")
    points_4d = load_points("ClusteringFiles/clusters_2clusts_4d_example02.csv", dim=4)
    clusters_4d = h_clustering(dim=4, k=2, points=points_4d)  # k=2 comme indiqué dans le nom du fichier
    save_points(
        clusters_4d,
        "resultats_4d.csv",
        "resultats_4d_tagged.csv"
    )
    print(f"Nombre de clusters 4D: {len(clusters_4d)}")
    for i, cluster in enumerate(clusters_4d):
        print(f"Cluster {i}: {len(cluster)} points")
    
    print("\nTests terminés. Fichiers de résultats générés :"
          "\n- resultats_3d.csv et resultats_3d_tagged.csv"
          "\n- resultats_6d.csv et resultats_6d_tagged.csv"
          "\n- resultats_4d.csv et resultats_4d_tagged.csv")
