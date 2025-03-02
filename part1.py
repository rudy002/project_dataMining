import numpy as np
import csv
import os
import math

def generate_data(dim, k, n, out_path, points_gen=None, extras={}):
    """
    Génère n points de dimension dim répartis en k clusters et les sauvegarde dans un fichier CSV.
    
    Paramètres:
    -----------
    dim : int
        Dimension des points à générer
    k : int
        Nombre de clusters à générer
    n : int
        Nombre total de points à générer
    out_path : str
        Chemin où sauvegarder le fichier CSV
    points_gen : function, optional
        Fonction optionnelle pour générer des points
    extras : dict, optional
        Paramètres supplémentaires
    """
    # Paramètres par défaut pour la génération des clusters
    std_dev = extras.get('std_dev', 0.5)
    max_coord = extras.get('max_coord', 10.0)
    min_points_per_cluster = extras.get('min_points_per_cluster', n // (2 * k))
    
    # Assurons-nous que le répertoire de sortie existe
    dirname = os.path.dirname(out_path)
    if dirname:  # Ne créer le répertoire que s'il y a un chemin de répertoire
        os.makedirs(dirname, exist_ok=True)
    
    # Si une fonction de génération de points est fournie, l'utiliser
    if points_gen:
        points = points_gen(dim, k, n, extras)
    else:
        # Générer nos propres points répartis en k clusters
        points = []
        
        # Distribuer les points entre les clusters (au moins min_points_per_cluster par cluster)
        points_per_cluster = [min_points_per_cluster] * k
        remaining_points = n - sum(points_per_cluster)
        
        # Distribuer les points restants de manière aléatoire entre les clusters
        for _ in range(remaining_points):
            points_per_cluster[np.random.randint(0, k)] += 1
            
        # Générer des centres de clusters aléatoires
        cluster_centers = []
        for _ in range(k):
            center = np.random.uniform(0, max_coord, dim)
            cluster_centers.append(center)
            
        # Générer des points autour de chaque centre
        for cluster_id, (center, num_points) in enumerate(zip(cluster_centers, points_per_cluster)):
            for _ in range(num_points):
                # Générer un point avec une distribution normale autour du centre
                point = center + np.random.normal(0, std_dev, dim)
                # Garder l'information du cluster dans la liste en mémoire, mais elle ne sera pas écrite dans le fichier
                points.append((tuple(point), cluster_id))
    
    # Écrire les points dans un fichier CSV (sans l'ID du cluster)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point, _ in points:  # Ignorer l'ID du cluster lors de l'écriture
            # Écrire seulement les coordonnées du point
            writer.writerow(list(point))
    
    print(f"Données générées et sauvegardées dans {out_path}: {n} points en {dim} dimensions répartis en {k} clusters (sans IDs de clusters)")
    return points

def load_points(in_path, dim, n=-1, points=None):
    """
    Charge des points depuis un fichier CSV.
    
    Paramètres:
    -----------
    in_path : str
        Chemin vers le fichier CSV à lire
    dim : int
        Dimension des points à lire
    n : int, optional
        Nombre maximum de points à lire (-1 pour tous les points)
    points : list, optional
        Liste où ajouter les points lus
    
    Retourne:
    ---------
    list
        Liste des points lus depuis le fichier
    """
    # Créer une nouvelle liste si points est None
    if points is None:
        points = []
    
    try:
        with open(in_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            
            for row in reader:
                if n != -1 and count >= n:
                    break
                    
                # Extraire les dim premières valeurs comme coordonnées du point
                if len(row) >= dim:
                    point_coords = tuple(float(val) for val in row[:dim])
                    points.append(point_coords)
                    count += 1
    
        print(f"Chargé {len(points)} points depuis {in_path}")
        return points
    except Exception as e:
        print(f"Erreur lors du chargement des points depuis {in_path}: {e}")
        return points

def euclidean_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.
    
    Paramètres:
    -----------
    point1, point2 : tuple
        Points dont on veut calculer la distance
    
    Retourne:
    ---------
    float
        Distance euclidienne entre les deux points
    """
    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir la même dimension")
    
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def h_clustering(dim, k, points, dist=None, clusts=None):
    """
    Effectue un clustering hiérarchique ascendant (bottom-up) sur un ensemble de points.
    
    Paramètres:
    -----------
    dim : int
        Dimension des points
    k : int
        Nombre final de clusters à obtenir
    points : list
        Liste des points à regrouper en clusters
    dist : function, optional
        Fonction de distance entre clusters
        Si None, utilise la distance minimale entre points (single linkage)
    clusts : list, optional
        Liste initiale de clusters
        Si None, chaque point commence dans son propre cluster
    
    Retourne:
    ---------
    list
        Liste de k clusters où chaque cluster est une liste de points
    """
    # Vérifier les entrées
    if not points:
        return []
    
    if k <= 0 or k > len(points):
        raise ValueError(f"Le nombre de clusters k doit être entre 1 et {len(points)}")
    
    # Si aucune fonction de distance n'est fournie, utiliser la distance minimale (single linkage)
    if dist is None:
        def dist(cluster1, cluster2):
            min_dist = float('inf')
            for p1 in cluster1:
                for p2 in cluster2:
                    d = euclidean_distance(p1, p2)
                    if d < min_dist:
                        min_dist = d
            return min_dist
    
    # Initialiser les clusters: si clusts est None, chaque point est son propre cluster
    if clusts is None:
        clusts = [[point] for point in points]
    
    # Si nous avons déjà le bon nombre de clusters, retourner directement
    if len(clusts) <= k:
        return clusts
    
    # Algorithme principal de clustering hiérarchique
    while len(clusts) > k:
        min_dist = float('inf')
        merge_i, merge_j = 0, 0
        
        # Trouver les deux clusters les plus proches
        for i in range(len(clusts)):
            for j in range(i + 1, len(clusts)):
                d = dist(clusts[i], clusts[j])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j
        
        # Fusionner les deux clusters les plus proches
        clusts[merge_i].extend(clusts[merge_j])
        clusts.pop(merge_j)
    
    return clusts

def k_means(dim, k, n, points, clusts=None):
    """
    Effectue un clustering K-means sur un ensemble de points.
    
    Paramètres:
    -----------
    dim : int
        Dimension des points
    k : int
        Nombre de clusters à former
    n : int
        Paramètre non utilisé, inclus pour compatibilité
    points : list
        Liste des points à regrouper en clusters
    clusts : list, optional
        Liste initiale de clusters (si fournie, doit contenir k clusters)
    
    Retourne:
    ---------
    list
        Liste de k clusters où chaque cluster est une liste de points
    """
    # Vérifier les entrées
    if not points:
        return []
    
    if k <= 0 or k > len(points):
        raise ValueError(f"Le nombre de clusters k doit être entre 1 et {len(points)}")
    
    # Convertir les points en tableaux numpy pour faciliter les calculs
    points_np = np.array(points)
    
    # Initialiser les centroides
    if clusts is not None:
        # Si des clusters sont fournis, calculer leurs centroides
        if len(clusts) != k:
            raise ValueError(f"Le nombre de clusters fournis ({len(clusts)}) doit être égal à k ({k})")
        
        centroids = np.array([np.mean(cluster, axis=0) for cluster in clusts])
    else:
        # Sinon, choisir k points aléatoires comme centroides initiaux
        indices = np.random.choice(len(points), k, replace=False)
        centroids = points_np[indices]
    
    # Initialiser les clusters
    clusters = [[] for _ in range(k)]
    
    # Variables pour suivre la convergence
    max_iterations = 100
    tolerance = 1e-4
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iterations:
        # Réinitialiser les clusters
        clusters = [[] for _ in range(k)]
        
        # Assigner chaque point au cluster du centroide le plus proche
        for i, point in enumerate(points_np):
            # Calculer la distance à chaque centroide
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            # Assigner au cluster le plus proche
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(tuple(point))
        
        # Sauvegarder les anciens centroides pour vérifier la convergence
        old_centroids = centroids.copy()
        
        # Mettre à jour les centroides
        for i in range(k):
            if clusters[i]:  # Éviter la division par zéro si un cluster est vide
                centroids[i] = np.mean(clusters[i], axis=0)
        
        # Vérifier la convergence
        centroid_shifts = [euclidean_distance(centroids[i], old_centroids[i]) for i in range(k)]
        total_shift = sum(centroid_shifts)
        
        if total_shift < tolerance:
            converged = True
            print(f"K-means a convergé après {iteration+1} itérations")
        
        iteration += 1
    
    if iteration == max_iterations:
        print(f"K-means a atteint le nombre maximum d'itérations ({max_iterations})")
    
    return clusters

def save_points(clusts, out_path, out_path_tagged):
    """
    Sauvegarde les clusters dans deux fichiers CSV.
    
    Paramètres:
    -----------
    clusts : list
        Liste de clusters où chaque cluster est une liste de points
    out_path : str
        Chemin où sauvegarder le fichier CSV sans les identifiants de clusters
    out_path_tagged : str
        Chemin où sauvegarder le fichier CSV avec les identifiants de clusters
    """
    # Créer les répertoires de sortie si nécessaire
    for path in [out_path, out_path_tagged]:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    
    # Sauvegarder les points sans identifiants de clusters
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster in enumerate(clusts):
            for point in cluster:
                # Écrire seulement les coordonnées du point
                writer.writerow(list(point))
    
    # Sauvegarder les points avec identifiants de clusters
    with open(out_path_tagged, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster in enumerate(clusts):
            for point in cluster:
                # Écrire les coordonnées et l'ID du cluster
                writer.writerow(list(point) + [cluster_id])
    
    # Compter le nombre total de points
    total_points = sum(len(cluster) for cluster in clusts)
    
    print(f"Sauvegardé {total_points} points dans {out_path} (sans identifiants de clusters)")
    print(f"Sauvegardé {total_points} points dans {out_path_tagged} (avec identifiants de clusters)")

# Exemple d'utilisation:
if __name__ == "__main__":
    # Exemple de génération de données
    generate_data(dim=3, k=3, n=100, out_path="example_data.csv")
    
    # Exemple de chargement de données
    points = []
    load_points(in_path="example_data.csv", dim=3, points=points)
    print(f"Nombre de points chargés: {len(points)}")
    
    # Exemple de clustering hiérarchique
    if points:
        clusters_h = h_clustering(dim=3, k=3, points=points)
        print(f"Nombre de clusters créés (hiérarchique): {len(clusters_h)}")
        for i, cluster in enumerate(clusters_h):
            print(f"Cluster {i}: {len(cluster)} points")
        
        # Exemple de clustering K-means
        clusters_k = k_means(dim=3, k=3, n=len(points), points=points)
        print(f"Nombre de clusters créés (K-means): {len(clusters_k)}")
        for i, cluster in enumerate(clusters_k):
            print(f"Cluster {i}: {len(cluster)} points")
        
        # Exemple de sauvegarde des résultats
        save_points(clusters_k, "results_kmeans.csv", "results_kmeans_tagged.csv")
