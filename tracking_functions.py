# DENSITY BASED DETECTION/TRACKING

from sklearn.cluster import MeanShift
from dbscan import DBSCAN
from hdbscan import HDBSCAN
import numpy as np
import torch


# NO CONSTRAINTS
def cluster_features_to_instances(dense_features, foreground_mask, method='hdbscan', margin=1.5, min_cell_size=100):
    """
    Clusters pixel embeddings into discrete instance masks. Uses a density-based clustering algorithm (DBSCAN, MeanShift, or HDBSCAN) to group pixels based on their feature embeddings.
    
    Args:
        dense_features: Tensor or numpy array of shape (T, C, H, W)
        foreground_mask: Binary mask of shape (T, H, W) where 1=cell, 0=background.
        method: 'hdbscan' (recommended)  or 'dbscan' or 'meanshift'.
        margin: The distance margin used during your contrastive training.
        min_cell_size: The minimum size of a cell in pixels.
        
    Returns:
        instance_mask: Numpy array of shape (T, H, W) with unique integer IDs for cells.
    """
    # 1. Convert to numpy
    if isinstance(dense_features, torch.Tensor):
        dense_features = dense_features.detach().cpu().squeeze(0).numpy()
    if isinstance(foreground_mask, torch.Tensor):
        foreground_mask = foreground_mask.detach().cpu().squeeze(0).numpy()


    T, C, H, W = dense_features.shape
    
    # 2. Flatten features: (H*W, C)
    features_flat = np.moveaxis(dense_features, 1, -1).reshape(-1, C)
    fg_mask_flat = foreground_mask.flatten() > 0
    
    # 3. Extract only foreground pixels to cluster
    # This drastically speeds up clustering and prevents background noise
    fg_features = features_flat[fg_mask_flat]
    
    # Create an empty instance mask (0 = background)
    instance_mask_flat = np.zeros(T*H * W, dtype=np.int32)
    
    # Edge case: No foreground detected
    if len(fg_features) == 0:
        return instance_mask_flat.reshape(T, H, W)

    labels = None  # Initialize labels variable
    # 4. Configure the Clustering Algorithm
    if method == 'dbscan':
        # eps should be slightly less than your training margin. 
        # If your margin was 1.5, points within the same cluster should be 
        # very close (e.g., < 0.5), while different clusters are > 1.5 aparigrapht.
        eps_threshold = margin * 0.2
        
        # min_samples prevents tiny noise dots from becoming cells.
        # Set this based on your minimum expected cell size (in pixels).
        #SKLEARN
        # clusterer = DBSCAN(eps=eps_threshold, min_samples=min_cell_size*T, metric='euclidean')
        #DBSCAN
        print(fg_features.shape)
        labels, core_samples = DBSCAN(fg_features, eps=eps_threshold, min_samples=min_cell_size*T)
    elif method == 'meanshift':
        # MeanShift bandwidth is also tied to your margin
        clusterer = MeanShift(bandwidth=margin * 0.5, bin_seeding=True)
    
    elif method == 'hdbscan':
        # HDBSCAN is a density-based clustering algorithm that can handle variable density clusters.
        # The min_cluster_size parameter controls the smallest size of clusters to consider.
        clusterer = HDBSCAN(min_cluster_size=min_cell_size*T, metric='euclidean')
        
    elif method == 'leiden':
        import leidenalg as la
        import igraph as ig
        # Leiden is a graph-based clustering algorithm. It requires constructing a k-nearest neighbor graph.
        # This is more complex and may require additional preprocessing.
        raise NotImplementedError("Leiden clustering is not yet implemented in this function.")
        
    else:
        raise ValueError("Method must be 'dbscan' or 'meanshift' or 'hdbscan'")

    # 5. Run Clustering
    print(f"Clustering {len(fg_features)} foreground pixels using {method.upper()}...")
    if labels is None:
        labels = clusterer.fit_predict(fg_features)
    
    # DBSCAN returns -1 for noise/outliers. We want to shift our IDs so they 
    # start at 1, keeping 0 exclusively for the background.
    if method == 'dbscan':
        # Map noise (-1) to background (0), and shift other clusters up by 1
        labels = np.where(labels == -1, -1, labels) + 1
    else:
        # MeanShift returns labels 0, 1, 2... Shift up by 1.
        labels = labels + 1

    # 6. Reconstruct the 2D Instance Mask
    # Place the cluster labels back into the original spatial locations
    instance_mask_flat[fg_mask_flat] = labels
    instance_mask_2d = instance_mask_flat.reshape(T, H, W)
    print(f"Detected {len(np.unique(instance_mask_2d)) - 1} instances.")
    return instance_mask_2d


#CONTRAINTE DE CONNECTED COMPONENTS

def dbscan_with_connected_components(dense_features, foreground_mask, eps=0.5, min_samples=2):
    """
    A custom DBSCAN that enforces a connected components constraint:
    Two pixels cannot belong to the same cluster if they are not spatially connected.
    
    Args:
        dense_features: (N, C) array of averaged 64-d cell embeddings.
        foreground_mask: (H, W) binary mask where 1=cell, 0=background.
        eps: Maximum feature distance to be considered the same cluster.
        min_samples: Minimum cluster size (in pixels) to be kept.
        
    Returns:
        instance_mask: (H, W) array of unique integer IDs for cells.
    """
    from scipy.ndimage import label

    if isinstance(dense_features, torch.Tensor):
        dense_features = dense_features.detach().cpu().squeeze().numpy()
    if isinstance(foreground_mask, torch.Tensor):
        foreground_mask = foreground_mask.detach().cpu().squeeze().numpy()


    C, H, W = dense_features.shape

    features_flat = np.moveaxis(dense_features, 1, -1).reshape(-1, C)
    
    # Step 1: Identify connected components in the foreground mask
    structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
    labeled_mask, num_features = label(foreground_mask, structure=structure)
    
    instance_mask = np.zeros_like(foreground_mask, dtype=np.int32)
    current_instance_id = 1
    
    for component_id in range(1, num_features + 1):
        component_mask = labeled_mask == component_id
        component_indices = np.argwhere(component_mask)
        if len(component_indices) < min_samples:
            continue  # Skip small components
        
        # Extract features for this component

        
        fg_mask_flat = component_mask.flatten() > 0
            
        fg_features = features_flat[fg_mask_flat]
        # Step 2: Apply DBSCAN within this connected component
        labels, core_samples = DBSCAN(fg_features, eps=eps, min_samples=min_samples)
        labels = np.where(labels == -1, -1, labels) + 1  # Shift labels to start from 1

        # Step 3: Assign unique instance IDs to the detected clusters
        for label_id in np.unique(labels):
            if label_id == -1:
                continue  # Skip noise
            
            cluster_mask = (labels == label_id)
            instance_mask[component_indices[cluster_mask][:, 0], component_indices[cluster_mask][:, 1]] = current_instance_id
            current_instance_id += 1
            
    return instance_mask



# GRAPH BASED DETECTION/TRACKING


from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx
from networkx.algorithms import community

import numpy as np
import torch
import networkx as nx

def build_spatial_feature_graph_nx(
    dense_features, 
    binary_mask, 
    spatial_connectivity=8, 
    affinity_sigma=1.0,
    distance_threshold=0.3
):
    """
    Constructs a 2D spatial-adjacency graph for a single frame using NetworkX.
    
    Args:
        dense_features: Tensor or array of shape (C, H, W)
        binary_mask: Tensor or array of shape (H, W) where 1=target pixel
        spatial_connectivity: 4 or 8 for 2D spatial connectivity
        affinity_sigma: Bandwidth for RBF kernel to convert distance to similarity
        distance_threshold: Edges with distance greater than this are culled 
                            before graph construction to save memory.
                            
    Returns:
        g: networkx.Graph object
        node_grid: (H, W) mapping from spatial coordinates to node IDs
    """
    # 1. Ensure inputs are NumPy arrays
    if isinstance(dense_features, torch.Tensor):
        dense_features = dense_features.detach().cpu().numpy()
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.detach().cpu().numpy()

    if dense_features.ndim != 3 or binary_mask.ndim != 2:
        raise ValueError(f"Expected (C, H, W) and (H, W), got {dense_features.shape} and {binary_mask.shape}")

    C, H, W = dense_features.shape
    fg_bool = binary_mask > 0

    # 2. Map target pixels to node IDs (0 to N-1)
    node_grid = np.full((H, W), -1, dtype=np.int32)
    node_indices = np.argwhere(fg_bool) # (N, 2) -> [y, x]
    N = len(node_indices)

    # Fast-fail if no foreground pixels exist
    if N == 0:
        return nx.Graph(), node_grid

    node_grid[fg_bool] = np.arange(N, dtype=np.int32)

    # Reshape features from (C, H, W) -> (H, W, C) -> extract (N, C)
    fg_features = np.moveaxis(dense_features, 0, -1)[fg_bool].astype(np.float32)

    # 3. Define 2D directional neighbor offsets
    offsets = [(0, 1), (1, 0)] # Down, Right
    if spatial_connectivity == 8:
        offsets.extend([(1, 1), (1, -1)]) # Diagonals: Down-Right, Down-Left

    # 4. Vectorized spatial edge extraction
    src_nodes_list, dst_nodes_list = [], []

    for dy, dx in offsets:
        y_src = slice(max(0, -dy), min(H, H - dy))
        x_src = slice(max(0, -dx), min(W, W - dx))
        
        y_dst = slice(max(0, dy), min(H, H + dy))
        x_dst = slice(max(0, dx), min(W, W + dx))
        
        src_ids = node_grid[y_src, x_src]
        dst_ids = node_grid[y_dst, x_dst]
        
        # Keep only pairs where both source and dest are valid foreground pixels
        valid = (src_ids >= 0) & (dst_ids >= 0)
        
        src_nodes_list.append(src_ids[valid])
        dst_nodes_list.append(dst_ids[valid])

    if len(src_nodes_list) == 0:
        # Edge case: No adjacent pixels
        g = nx.Graph()
        g.add_nodes_from(range(N))
        return g, node_grid

    src_indices = np.concatenate(src_nodes_list)
    dst_indices = np.concatenate(dst_nodes_list)

    # 5. Calculate Feature Distances
    diffs = fg_features[src_indices] - fg_features[dst_indices]
    distances = np.linalg.norm(diffs, axis=1)
    
    # 6. Apply pre-filtering (Massive optimization for NetworkX)
    if distance_threshold is not None:
        valid_mask = distances <= distance_threshold
        src_indices = src_indices[valid_mask]
        dst_indices = dst_indices[valid_mask]
        distances = distances[valid_mask]
        
    affinities = np.exp(-distances / (affinity_sigma + 1e-8))

    # 7. Build the NetworkX graph in bulk
    g = nx.Graph()
    g.add_nodes_from(range(N))
    
    # NetworkX expects edges as (u, v, attribute_dict)
    # Using a list comprehension is the fastest way to package this for NetworkX
    edge_attributes = [{'distance': d, 'weight': w} for d, w in zip(distances.tolist(), affinities.tolist())]
    edges_with_attrs = zip(src_indices.tolist(), dst_indices.tolist(), edge_attributes)
    
    g.add_edges_from(edges_with_attrs)
    
    return g, node_grid

def cluster_spatial_graph(g, node_grid, binary_mask, min_cell_size=50, resolution=0.0000001):
    """
    Step 2: Uses Louvain to cluster the spatial graph into individual cells.
    """
    if g.number_of_nodes() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)
        
    communities = community.louvain_communities(g, weight='weight', resolution=resolution)
    
    node_to_cluster_map = np.zeros(g.number_of_nodes(), dtype=np.int32)
    cluster_id = 1
    
    for node_set in communities:
        if len(node_set) >= min_cell_size:
            node_to_cluster_map[list(node_set)] = cluster_id
            cluster_id += 1
            
    instance_mask = np.zeros_like(binary_mask, dtype=np.int32)
    valid_pixels = node_grid >= 0
    instance_mask[valid_pixels] = node_to_cluster_map[node_grid[valid_pixels]]
    
    return instance_mask



#TRACKING FUNCTIONS

from scipy.ndimage import center_of_mass
def instance_mask_to_features(instance_mask, dense_features):
    """
    Converts an instance mask and dense features into a feature vector for each instance.
    
    Args:
        instance_mask: 2D array of shape (H, W) with unique integer IDs for instances.
        dense_features: 3D array of shape (C, H, W) representing dense features.
    Returns:
        instance_features: (N, C) array of averaged 64-d cell embeddings.
        loc: (N, 2) array of cell centroids in pixel coordinates.
    """

    
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
    
    instance_features = []
    loc = []
    for inst_id in unique_ids:
        mask = instance_mask == inst_id
        if np.sum(mask) == 0:
            continue
        # Average the dense features over the pixels belonging to this instance
        avg_feature = dense_features[:, mask].mean(axis=1)
        instance_features.append(avg_feature)
        # Calculate the centroid of the instance
        centroid = center_of_mass(mask)
        loc.append(centroid)
    
    return np.array(instance_features), np.array(loc)


import numpy as np
from scipy.spatial.distance import cdist

def constrained_dbscan_tracking(features, frames, eps=0.5, min_samples=2): #buggy ?
    """
    Custom DBSCAN that enforces a Cannot-Link constraint:
    Two points cannot belong to the same cluster if they share the same frame ID.
    
    Args:
        features: (N, C) array of averaged 64-d cell embeddings.
        frames: (N,) integer array of the frame ID for each cell.
        eps: Maximum feature distance to be considered the same track.
        min_samples: Minimum track length (in frames) to be kept.
        
    Returns:
        labels: (N,) array of Track IDs. -1 indicates noise/untracked.
    """
    N = len(features)
    labels = np.full(N, -1, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    
    # Precompute all pairwise distances (Fast for a few thousand cells)
    dist_matrix = cdist(features, features, metric='euclidean')
    
    cluster_id = 1
    
    for p_idx in range(N):
        if visited[p_idx]:
            continue
            
        visited[p_idx] = True
        
        # 1. Region Query: Find all points within eps distance
        neighbors = np.where(dist_matrix[p_idx] <= eps)[0]
        
        # 2. Hard Constraint: Remove neighbors in the exact same frame as the seed
        neighbors = [n for n in neighbors if frames[n] != frames[p_idx]]
        
        # Check density (Core point)
        if len(neighbors) + 1 < min_samples:
            continue # Remains -1 (Noise)
            
        # 3. Initialize new track
        labels[p_idx] = cluster_id
        frames_in_cluster = {frames[p_idx]}
        
        # Sort neighbors by distance to the seed point! 
        # This ensures the closest feature match claims the frame slot first.
        neighbors = sorted(neighbors, key=lambda n: dist_matrix[p_idx, n])
        queue = list(neighbors)
        
        # 4. Expand Cluster
        while queue:
            q_idx = queue.pop(0)
            
            if not visited[q_idx]:
                visited[q_idx] = True
                q_neighbors = np.where(dist_matrix[q_idx] <= eps)[0]
                q_neighbors = [n for n in q_neighbors if frames[n] != frames[q_idx]]
                
                if len(q_neighbors) + 1 >= min_samples:
                    # Sort expansion candidates by distance to q_idx
                    q_neighbors = sorted(q_neighbors, key=lambda n: dist_matrix[q_idx, n])
                    queue.extend(q_neighbors)
                    
            # 5. Apply Cannot-Link Constraint during assignment
            if labels[q_idx] == -1:
                if frames[q_idx] not in frames_in_cluster:
                    labels[q_idx] = cluster_id
                    frames_in_cluster.add(frames[q_idx])
                    
        cluster_id += 1

    return labels