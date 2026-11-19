import concurrent.futures as cf
from skimage.measure import label
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, MeanShift

import torch
import numpy as np
from skimage.measure import regionprops

def cluster_features_to_instances(dense_features, foreground_mask, method='dbscan', margin=1.5, min_cell_size=100):
    """
    Clusters 64-d pixel embeddings into discrete instance masks.
    
    Args:
        dense_features: Tensor or numpy array of shape (C, H, W)
        foreground_mask: Binary mask of shape (H, W) where 1=cell, 0=background.
                         (Usually predicted by a parallelfirst_try/SiameseUNet/version_1/checkpoints/last.ckpt semantic branch).
        method: 'dbscan' (recommended) or 'meanshift'.
        margin: The distance margin used during your contrastive training.
        min_cell_size: The minimum size of a cell in pixels.
        
    Returns:
        instance_mask: Numpy array of shape (H, W) with unique integer IDs for cells.
    """
    # 1. Convert to numpy
    if isinstance(dense_features, torch.Tensor):
        dense_features = dense_features.detach().cpu().squeeze().numpy()
    if isinstance(foreground_mask, torch.Tensor):
        foreground_mask = foreground_mask.detach().cpu().squeeze().numpy()


    C, H, W = dense_features.shape
    
    # 2. Flatten features: (H*W, C)
    features_flat = dense_features.reshape(C, -1).T
    fg_mask_flat = foreground_mask.flatten() > 0
    
    # 3. Extract only foreground pixels to cluster
    # This drastically speeds up clustering and prevents background noise
    fg_features = features_flat[fg_mask_flat]
    
    # Create an empty instance mask (0 = background)
    instance_mask_flat = np.zeros(H * W, dtype=np.int32)
    
    # Edge case: No foreground detected
    if len(fg_features) == 0:
        return instance_mask_flat.reshape(H, W)
        
    # 4. Configure the Clustering Algorithm
    if method == 'dbscan':
        # eps should be slightly less than your training margin. 
        # If your margin was 1.5, points within the same cluster should be 
        # very close (e.g., < 0.5), while different clusters are > 1.5 apart.
        eps_threshold = margin * 0.2
        
        # min_samples prevents tiny noise dots from becoming cells.
        # Set this based on your minimum expected cell size (in pixels).
        clusterer = DBSCAN(eps=eps_threshold, min_samples=min_cell_size, metric='euclidean')
        
    elif method == 'meanshift':
        # MeanShift bandwidth is also tied to your margin
        clusterer = MeanShift(bandwidth=margin * 0.5, bin_seeding=True)
    
    elif method == 'hdbscan':
        # HDBSCAN is a density-based clustering algorithm that can handle variable density clusters.
        # The min_cluster_size parameter controls the smallest size of clusters to consider.
        clusterer = HDBSCAN(min_cluster_size=min_cell_size, metric='euclidean')
        
    elif method == 'leiden':
        import leidenalg as la
        import igraph as ig
        # Leiden is a graph-based clustering algorithm. It requires constructing a k-nearest neighbor graph.
        # This is more complex and may require additional preprocessing.
        raise NotImplementedError("Leiden clustering is not yet implemented in this function.")
        
    else:
        raise ValueError("Method must be 'dbscan' or 'meanshift'")

    # 5. Run Clustering
    print(f"Clustering {len(fg_features)} foreground pixels using {method.upper()}...")
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
    instance_mask_2d = instance_mask_flat.reshape(H, W)
    print(f"Detected {len(np.unique(instance_mask_2d)) - 1} instances.")
    return instance_mask_2d


# foreground_mask = (sigmoid(preds[0][0]).detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)
# instance_mask = cluster_features_to_instances(preds[0][1], foreground_mask, method='dbscan', margin=1.5)
# unique_labels=np.unique(instance_mask)
# cmap = plt.cm.get_cmap('tab20', len(unique_labels)) 
   
# plt.figure(figsize=(10,10))
# plt.imshow(instance_mask, cmap=cmap)
# plt.colorbar()
# print(f"Number of detected instances: {len(np.unique(instance_mask)) - 1}")  # Subtract 1 for background
# print("GT number of instances:", len(np.unique(gt)) - 1)




def metrics_per_label(instance_mask, dense_features, include_background=False):
    """
    Computes the average feature vector, and the variance for each unique label in the instance mask.
    
    Args:
        instance_mask: 2D numpy array of shape (H, W) with unique integer IDs for instances.
        dense_features: 3D numpy array of shape (C, H, W) representing pixel embeddings.
        
    Returns:
        metrics: Dictionary mapping each label to a tuple of (average_feature_vector, variance).
    """
    unique_labels = np.unique(instance_mask)
    avg_features = {}
    var_features = {}
    for label in unique_labels:
        if label == 0:  # Skip background
            if include_background:
                avg_vector = dense_features.squeeze()[:,instance_mask == label].mean(axis=1)
                var_vector = dense_features.squeeze()[:,instance_mask == label].var(axis=1)
                
            else:
                continue
        mask = (instance_mask == label)
        if np.sum(mask) == 0:
            continue
        avg_vector = dense_features.squeeze()[:, mask].mean(axis=1)
        var_vector = dense_features.squeeze()[:, mask].var(axis=1).norm()

        avg_features[label] = avg_vector
        var_features[label] = var_vector
    #normalize variance to be between 0 and 1
    var_values = np.array(list(var_features.values()))
    if len(var_values) > 0:
        min_var = var_values.min()
        max_var = var_values.max()
        for label in var_features:
            if max_var - min_var > 0:
                var_features[label] = (var_features[label] - min_var) / (max_var - min_var)
            else:
                var_features[label] = 0.0
    
    return avg_features, var_features





def cluster_features_to_instances_connected_components(dense_features, foreground_mask, min_cell_size=100, margin=1.5, method='dbscan', num_workers=1):
    """
    Begins by running connected components on the foreground mask to get initial instance masks. 
    Then, for each connected component, it clusters the features within that component to potentially split it into multiple instances.

    The connected components are iterated with `regionprops`, and each component is cropped to its
    bounding box before clustering to reduce per-component work.

    Args:
        num_workers: Number of concurrent workers to use for per-component clustering.
            Set to 1 to run single-threaded.
    """

    # 1. Convert to numpy
    if isinstance(dense_features, torch.Tensor):
        dense_features = dense_features.detach().cpu().squeeze().numpy()
    if isinstance(foreground_mask, torch.Tensor):
        foreground_mask = foreground_mask.detach().cpu().squeeze().numpy()

    C, H, W = dense_features.shape
    
    # 2. Run connected components on the foreground mask
    initial_instance_mask = label(foreground_mask, connectivity=1)  
    
    # 3. Prepare the final instance mask
    final_instance_mask = np.zeros_like(initial_instance_mask, dtype=np.int32)
    current_label = 1
    
    def _cluster_component(prop):
        min_row, min_col, max_row, max_col = prop.bbox
        component_mask = prop.image
        component_area = prop.area

        if component_area < min_cell_size:
            return prop.label, None, None

        component_features = dense_features[:, min_row:max_row, min_col:max_col]
        print(f"Clustering component {prop.label} with {component_area} pixels using {method.upper()}...")
        clustered_instance_mask = cluster_features_to_instances(
            component_features,
            component_mask.astype(np.uint8),
            method=method,
            margin=margin,
            min_cell_size=min_cell_size,
        )
        return prop.label, (min_row, min_col, max_row, max_col), clustered_instance_mask

    # 4. For each connected component, cluster the features within it
    component_props = regionprops(initial_instance_mask)

    if num_workers and num_workers > 1 and len(component_props) > 1:
        with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
            component_results = list(executor.map(_cluster_component, component_props))
    else:
        component_results = [_cluster_component(prop) for prop in component_props]

    for label_id, bbox, clustered_instance_mask in component_results:
        if clustered_instance_mask is None:
            continue

        min_row, min_col, max_row, max_col = bbox
        component_mask = initial_instance_mask[min_row:max_row, min_col:max_col] == label_id
        final_component_view = final_instance_mask[min_row:max_row, min_col:max_col]

        # Update the final instance mask with new labels
        unique_cluster_labels = np.unique(clustered_instance_mask)
        for cluster_label in unique_cluster_labels:
            if cluster_label == 0:  # Skip background
                continue

            final_component_view[component_mask & (clustered_instance_mask == cluster_label)] = current_label
            current_label += 1
            
    return final_instance_mask