import numpy as np
import csv
import os

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

# Exemple d'utilisation:
if __name__ == "__main__":
    # Exemple de génération de données
    generate_data(dim=3, k=3, n=100, out_path="example_data.csv")
    
    # Exemple de chargement de données
    points = []
    load_points(in_path="example_data.csv", dim=3, points=points)
    print(f"Nombre de points chargés: {len(points)}")
