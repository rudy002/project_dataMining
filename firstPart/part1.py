import numpy as np
import csv
import os
import math

def generate_data(dim, k, n, out_path, points_gen=None, extras={}):
    """
    Generates n points of dimension dim distributed in k clusters and saves them to a CSV file.
    
    Parameters:
    -----------
    dim : int
        Dimension of points to generate
    k : int
        Number of clusters to generate
    n : int
        Total number of points to generate
    out_path : str
        Path where to save the CSV file
    points_gen : function, optional
        Optional function to generate points
    extras : dict, optional
        Additional parameters
    """
    # Default parameters for cluster generation
    std_dev = extras.get('std_dev', 0.5)
    max_coord = extras.get('max_coord', 10.0)
    min_points_per_cluster = extras.get('min_points_per_cluster', n // (2 * k))
    
    # Ensure output directory exists
    dirname = os.path.dirname(out_path)
    if dirname:  # Only create directory if there's a directory path
        os.makedirs(dirname, exist_ok=True)
    
    # If a point generation function is provided, use it
    if points_gen:
        points = points_gen(dim, k, n, extras)
    else:
        # Generate our own points distributed in k clusters
        points = []
        
        # Distribute points among clusters (at least min_points_per_cluster per cluster)
        points_per_cluster = [min_points_per_cluster] * k
        remaining_points = n - sum(points_per_cluster)
        
        # Distribute remaining points randomly among clusters
        for _ in range(remaining_points):
            points_per_cluster[np.random.randint(0, k)] += 1
            
        # Generate random cluster centers
        cluster_centers = []
        for _ in range(k):
            center = np.random.uniform(0, max_coord, dim)
            cluster_centers.append(center)
            
        # Generate points around each center
        for cluster_id, (center, num_points) in enumerate(zip(cluster_centers, points_per_cluster)):
            for _ in range(num_points):
                # Generate point with normal distribution around center
                point = center + np.random.normal(0, std_dev, dim)
                # Keep cluster information in memory list, but it won't be written to file
                points.append((tuple(point), cluster_id))
    
    # Create path for tagged data
    out_path_tagged = os.path.splitext(out_path)[0] + "_tagged.csv"
    
    # Write points to CSV file (without cluster ID)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point, _ in points:  # Ignore cluster ID when writing
            # Write only point coordinates
            writer.writerow(list(point))
    
    # Write points to CSV file (with cluster ID)
    with open(out_path_tagged, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point, cluster_id in points:
            # Write point coordinates and cluster ID
            writer.writerow(list(point) + [cluster_id])
    
    print(f"Data generated and saved to {out_path}: {n} points in {dim} dimensions distributed in {k} clusters (without cluster IDs)")
    print(f"Data generated and saved to {out_path_tagged}: {n} points in {dim} dimensions distributed in {k} clusters (with cluster IDs)")
    return points

def load_points(in_path, dim, n=-1, points=None):
    """
    Loads points from a CSV file.
    
    Parameters:
    -----------
    in_path : str
        Path to CSV file to read
    dim : int
        Dimension of points to read
    n : int, optional
        Maximum number of points to read (-1 for all points)
    points : list, optional
        List where to add read points
    
    Returns:
    --------
    list
        List of points read from file
    """
    # Create new list if points is None
    if points is None:
        points = []
    
    try:
        with open(in_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            
            for row in reader:
                if n != -1 and count >= n:
                    break
                    
                # Extract first dim values as point coordinates
                if len(row) >= dim:
                    point_coords = tuple(float(val) for val in row[:dim])
                    points.append(point_coords)
                    count += 1
    
        print(f"Loaded {len(points)} points from {in_path}")
        return points
    except Exception as e:
        print(f"Error loading points from {in_path}: {e}")
        return points

def euclidean_distance(point1, point2):
    """
    Calculates Euclidean distance between two points.
    
    Parameters:
    -----------
    point1, point2 : tuple
        Points to calculate distance between
    
    Returns:
    --------
    float
        Euclidean distance between the two points
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def h_clustering(dim, k, points, dist=None, clusts=None):
    """
    Performs hierarchical agglomerative (bottom-up) clustering on a set of points.
    
    Parameters:
    -----------
    dim : int
        Dimension of points
    k : int
        Final number of clusters to obtain
    points : list
        List of points to cluster
    dist : function, optional
        Distance function between clusters
        If None, uses minimum distance between points (single linkage)
    clusts : list, optional
        Initial list of clusters
        If None, each point starts in its own cluster
    
    Returns:
    --------
    list
        List of k clusters where each cluster is a list of points
    """
    # Check inputs
    if not points:
        return []
    
    if k <= 0 or k > len(points):
        raise ValueError(f"Number of clusters k must be between 1 and {len(points)}")
    
    # If no distance function provided, use minimum distance (single linkage)
    if dist is None:
        def dist(cluster1, cluster2):
            min_dist = float('inf')
            for p1 in cluster1:
                for p2 in cluster2:
                    d = euclidean_distance(p1, p2)
                    if d < min_dist:
                        min_dist = d
            return min_dist
    
    # Initialize clusters: if clusts is None, each point is its own cluster
    if clusts is None:
        clusts = [[point] for point in points]
    
    # If we already have the right number of clusters, return directly
    if len(clusts) <= k:
        return clusts
    
    # Main hierarchical clustering algorithm
    while len(clusts) > k:
        min_dist = float('inf')
        merge_i, merge_j = 0, 0
        
        # Find two closest clusters
        for i in range(len(clusts)):
            for j in range(i + 1, len(clusts)):
                d = dist(clusts[i], clusts[j])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j
        
        # Merge two closest clusters
        clusts[merge_i].extend(clusts[merge_j])
        clusts.pop(merge_j)
    
    return clusts

def k_means(dim, k, n, points, clusts=None):
    """
    Performs K-means clustering on a set of points.
    
    Parameters:
    -----------
    dim : int
        Dimension of points
    k : int
        Number of clusters to form
    n : int
        Unused parameter, included for compatibility
    points : list
        List of points to cluster
    clusts : list, optional
        Initial list of clusters (if provided, must contain k clusters)
    
    Returns:
    --------
    list
        List of k clusters where each cluster is a list of points
    """
    # Check inputs
    if not points:
        return []
    
    if k <= 0 or k > len(points):
        raise ValueError(f"Number of clusters k must be between 1 and {len(points)}")
    
    # Convert points to numpy arrays for easier calculations
    points_np = np.array(points)
    
    # Initialize centroids
    if clusts is not None:
        # If clusters provided, calculate their centroids
        if len(clusts) != k:
            raise ValueError(f"Number of provided clusters ({len(clusts)}) must equal k ({k})")
        
        centroids = np.array([np.mean(cluster, axis=0) for cluster in clusts])
    else:
        # Otherwise, choose k random points as initial centroids
        indices = np.random.choice(len(points), k, replace=False)
        centroids = points_np[indices]
    
    # Initialize clusters
    clusters = [[] for _ in range(k)]
    
    # Main K-means loop
    max_iterations = 100
    for iteration in range(max_iterations):
        # Clear current clusters
        for cluster in clusters:
            cluster.clear()
        
        # Assign points to nearest centroid
        for point in points:
            distances = [np.sum((point - centroid) ** 2) for centroid in centroids]
            closest = np.argmin(distances)
            clusters[closest].append(point)
        
        # Update centroids
        old_centroids = centroids.copy()
        for i, cluster in enumerate(clusters):
            if cluster:  # Only update if cluster is not empty
                centroids[i] = np.mean(cluster, axis=0)
        
        # Check for convergence
        if np.allclose(old_centroids, centroids):
            break
    
    return clusters

def save_points(clusts, out_path, out_path_tagged):
    """
    Saves clusters to two CSV files.
    
    Parameters:
    -----------
    clusts : list
        List of clusters where each cluster is a list of points
    out_path : str
        Path where to save the CSV file without cluster IDs
    out_path_tagged : str
        Path where to save the CSV file with cluster IDs
    """
    # Create output directories if needed
    for path in [out_path, out_path_tagged]:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    
    # Save points without cluster IDs
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster in enumerate(clusts):
            for point in cluster:
                # Write only point coordinates
                writer.writerow(list(point))
    
    # Save points with cluster IDs
    with open(out_path_tagged, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster in enumerate(clusts):
            for point in cluster:
                # Write coordinates and cluster ID
                writer.writerow(list(point) + [cluster_id])
    
    # Count total number of points
    total_points = sum(len(cluster) for cluster in clusts)
    
    print(f"Saved {total_points} points to {out_path} (without cluster IDs)")
    print(f"Saved {total_points} points to {out_path_tagged} (with cluster IDs)")

def calculate_accuracy(clusters, tagged_file_path, dim):
    """
    Calculates the accuracy of clusters in a simplified way.
    
    Parameters:
    -----------
    clusters : list
        List of clusters generated by the algorithm
    tagged_file_path : str
        Path to the CSV file containing tagged data
    dim : int
        Dimension of points
        
    Returns:
    --------
    float
        Approximate accuracy of clustering (between 0 and 1)
    """
    # Load points with their real tags
    points_with_tags = []
    try:
        with open(tagged_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= dim + 1:
                    point_coords = tuple(float(val) for val in row[:dim])
                    cluster_id = int(row[dim])
                    points_with_tags.append((point_coords, cluster_id))
    except Exception as e:
        print(f"Error loading tagged data: {e}")
        return 0.0
    
    # Create a dictionary to know which cluster each point belongs to
    point_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            point_to_cluster[point] = cluster_id
    
    # Count how many points from the same real cluster are in the same predicted cluster
    correct = 0
    total_pairs = 0
    
    # For each pair of points
    for i in range(len(points_with_tags)):
        point1, tag1 = points_with_tags[i]
        for j in range(i+1, len(points_with_tags)):
            point2, tag2 = points_with_tags[j]
            
            # Check if the two points are in the same real cluster
            same_real_cluster = (tag1 == tag2)
            
            # Check if the two points are in the same predicted cluster
            same_pred_cluster = False
            if point1 in point_to_cluster and point2 in point_to_cluster:
                same_pred_cluster = (point_to_cluster[point1] == point_to_cluster[point2])
            
            # If the prediction matches reality, it's correct
            if same_real_cluster == same_pred_cluster:
                correct += 1
            
            total_pairs += 1
    
    # Calculate accuracy
    accuracy = correct / total_pairs if total_pairs > 0 else 0.0
    return accuracy

# Example usage:
if __name__ == "__main__":
    # Example of data generation
    generate_data(dim=3, k=2, n=1000, out_path="firstPart/example_data.csv")
    
    # Example of data loading
    points = []
    load_points(in_path="firstPart/example_data_dim3_k2.csv", dim=3, points=points)
    print(f"Number of points loaded: {len(points)}")
    
    # Example of hierarchical clustering
    if points:
        clusters_h = h_clustering(dim=3, k=2, points=points)
        print(f"Number of clusters created (hierarchical): {len(clusters_h)}")
        for i, cluster in enumerate(clusters_h):
            print(f"Cluster {i}: {len(cluster)} points")
        
        # Calculate accuracy for hierarchical clustering
        h_accuracy = calculate_accuracy(clusters_h, "firstPart/example_data_dim3_k2_tagged.csv", dim=3)
        print(f"Accuracy of hierarchical clustering: {h_accuracy:.4f}")
        
        # Example of K-means clustering
        clusters_k = k_means(dim=3, k=2, n=len(points), points=points)
        print(f"Number of clusters created (K-means): {len(clusters_k)}")
        for i, cluster in enumerate(clusters_k):
            print(f"Cluster {i}: {len(cluster)} points")
        
        # Calculate accuracy for K-means clustering
        k_accuracy = calculate_accuracy(clusters_k, "firstPart/example_data_tagged.csv", dim=3)
        print(f"Accuracy of K-means: {k_accuracy:.4f}")
        
        # Example of saving results
        save_points(clusters_k, "firstPart/results_kmeans.csv", "firstPart/results_kmeans_tagged.csv")
