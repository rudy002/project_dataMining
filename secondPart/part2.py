import sys
import os
import numpy as np
import csv

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstPart.part1 import generate_data

def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    """
    Implements the BFR algorithm for clustering large datasets.
    
    Parameters:
    -----------
    dim : int
        Dimension of the points
    k : int
        Number of desired clusters
    n : int
        Number of points to process
    block_size : int
        Number of points processed in each block
    in_path : str
        Path to the input CSV file containing points
    out_path : str
        Path where to save the results
    """
    # Initialization
    clusters = []  # List of clusters
    points_processed = 0
    current_block = []
    
    # Read and process file by blocks
    with open(in_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            # Convert coordinates to float
            point = np.array([float(x) for x in row])
            current_block.append(point)
            
            # When block is full, process it
            if len(current_block) >= block_size:
                points = np.array(current_block)
                
                # If first block, initialize clusters
                if not clusters:
                    # Choose k random points as initial centers
                    indices = np.random.choice(len(points), k, replace=False)
                    for idx in indices:
                        clusters.append({
                            'center': points[idx],
                            'points': [],
                            'sum': np.zeros(dim),
                            'sum_sq': np.zeros(dim),
                            'n': 0
                        })
                
                # Assign points to clusters and update statistics
                for point in points:
                    # Find closest cluster
                    distances = [np.sum((point - c['center'])**2) for c in clusters]
                    closest = np.argmin(distances)
                    
                    # Update cluster statistics
                    cluster = clusters[closest]
                    cluster['points'].append(point)
                    cluster['sum'] += point
                    cluster['sum_sq'] += point**2
                    cluster['n'] += 1
                    
                    # Update center (mean)
                    cluster['center'] = cluster['sum'] / cluster['n']
                
                # Empty current block
                current_block = []
                points_processed += block_size
                print(f"Processed {points_processed}/{n} points")
    
    # Process remaining points
    if current_block:
        points = np.array(current_block)
        for point in points:
            distances = [np.sum((point - c['center'])**2) for c in clusters]
            closest = np.argmin(distances)
            
            cluster = clusters[closest]
            cluster['points'].append(point)
            cluster['sum'] += point
            cluster['sum_sq'] += point**2
            cluster['n'] += 1
            cluster['center'] = cluster['sum'] / cluster['n']
    
    # Save results
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for i, cluster in enumerate(clusters):
            for point in cluster['points']:
                writer.writerow(list(point) + [i])
    
    return clusters

def cure_cluster(dim, k, n, block_size, in_path, out_path):
    """
    Implements the CURE algorithm for clustering large datasets.
    
    Parameters:
    -----------
    dim : int
        Dimension of the points
    k : int
        Number of desired clusters (None if k should be determined)
    n : int
        Number of points to process
    block_size : int
        Number of points processed in each block
    in_path : str
        Path to the input CSV file containing points
    out_path : str
        Path where to save the results
    """
    # Initialization
    clusters = []  # List of clusters
    points_processed = 0
    current_block = []
    
    # Read and process file by blocks
    with open(in_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            # Convert coordinates to float
            point = np.array([float(x) for x in row])
            current_block.append(point)
            
            # When block is full, process it
            if len(current_block) >= block_size:
                points = np.array(current_block)
                
                # If first block and k is None, determine k
                if not clusters and k is None:
                    # Use simple heuristic for k
                    k = min(int(np.sqrt(len(points))), 10)
                
                # If first block, initialize clusters
                if not clusters:
                    # Choose k random points as initial centers
                    indices = np.random.choice(len(points), k, replace=False)
                    for idx in indices:
                        clusters.append({
                            'center': points[idx],
                            'points': []
                        })
                
                # Assign points to clusters
                for point in points:
                    # Find closest cluster
                    distances = [np.sum((point - c['center'])**2) for c in clusters]
                    closest = np.argmin(distances)
                    clusters[closest]['points'].append(point)
                
                # Update centers
                for cluster in clusters:
                    if cluster['points']:
                        cluster['center'] = np.mean(cluster['points'], axis=0)
                
                # Empty current block
                current_block = []
                points_processed += block_size
                print(f"Processed {points_processed}/{n} points")
    
    # Process remaining points
    if current_block:
        points = np.array(current_block)
        for point in points:
            distances = [np.sum((point - c['center'])**2) for c in clusters]
            closest = np.argmin(distances)
            clusters[closest]['points'].append(point)
    
    # Save results
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for i, cluster in enumerate(clusters):
            for point in cluster['points']:
                writer.writerow(list(point) + [i])
    
    return clusters

if __name__ == "__main__":
    # Generate test dataset
    test_points = generate_data(
        dim=2,
        k=5,
        n=10000,
        out_path="firstPart/example_data.csv"
    )
    print("\nTest data generation completed.")
    
    # Test BFR algorithm
    print("\nTesting BFR algorithm...")
    clusters_bfr = bfr_cluster(
        dim=2,
        k=5,
        n=10000,
        block_size=1000,
        in_path="firstPart/example_data.csv",
        out_path="secondPart/bfr_results.csv"
    )
    print("\nBFR clustering completed.")
    print(f"Number of clusters: {len(clusters_bfr)}")
    for i, cluster in enumerate(clusters_bfr):
        print(f"Cluster {i}: {cluster['n']} points")
    
    # Test CURE algorithm
    print("\nTesting CURE algorithm...")
    clusters_cure = cure_cluster(
        dim=2,
        k=5,
        n=10000,
        block_size=1000,
        in_path="secondPart/test_data.csv",
        out_path="secondPart/cure_results.csv"
    )
    print("\nCURE clustering completed.")
    print(f"Number of clusters: {len(clusters_cure)}")
    for i, cluster in enumerate(clusters_cure):
        print(f"Cluster {i}: {len(cluster['points'])} points")
    
    print("\nResults have been saved to:")
    print("- BFR results: secondPart/bfr_results.csv")
    print("- CURE results: secondPart/cure_results.csv")

