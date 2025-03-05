import numpy as np

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
            
        # Afficher la progression
        if (cluster_id + 1) % 10 == 0 or cluster_id == k - 1:
            print(f"Généré {cluster_id + 1}/{k} clusters")
    
    return points 