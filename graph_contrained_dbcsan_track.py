import napari
import tifffile
from tracking_functions import build_spatial_feature_graph_nx, cluster_spatial_graph, instance_mask_to_features, constrained_dbscan_tracking, dbscan_with_connected_components
import pickle as pkl
import numpy as np
from tqdm import tqdm

n_tested=10


preds=pkl.load(open("./results/Fluo-N2DH-GOWT1/5_frames/3d_preds_v2_best.pkl", "rb"))
foreground_mask, dense_features = preds[0][0].squeeze(0) > 0.5, preds[0][1].squeeze(0)
for i in range(1,n_tested):
    foreground_mask = np.concatenate((foreground_mask, preds[i][0].squeeze(0) > 0.5), axis=0)
    dense_features = np.concatenate((dense_features, preds[i][1].squeeze(0)), axis=0)
print(foreground_mask.shape, dense_features.shape)



#graph based segmentation
# instance_masks = np.zeros_like(foreground_mask, dtype=np.int32)
# for i in range(n_tested):
#     g_spatial, node_grid = build_spatial_feature_graph_nx(
#         dense_features[i], 
#         foreground_mask[i], 
#         distance_threshold=1.5)
#     print(f"Frame {i}: Graph has {g_spatial.number_of_nodes()} nodes and {g_spatial.number_of_edges()} edges.")
#     instance_mask = cluster_spatial_graph(
#         g_spatial, 
#         node_grid, 
#         foreground_mask[i],
#         min_cell_size=1000,
#         resolution=0.0000001
#     )
#     print(f"Frame {i}: Detected {len(np.unique(instance_mask)) - 1} instances.")
#     instance_masks[i] = instance_mask


#density based segmentation
instance_masks = np.zeros_like(foreground_mask, dtype=np.int32)
for i in range(n_tested):
    instance_mask = dbscan_with_connected_components(
        dense_features[i], 
        foreground_mask[i],
        eps=10,
        min_samples=1000
    )
    print(instance_mask.shape)
    print(f"Frame {i}: Detected {len(np.unique(instance_mask)) - 1} instances.")
    print(f"Frame {i}: Unique instance IDs: {np.unique(instance_mask)}")
    instance_masks[i] = instance_mask

instance_features=np.empty((0, dense_features.shape[1]), dtype=np.float32)
frames=[]
inst_locs=np.empty((0, 2), dtype=np.float32)
for i in range(n_tested):
    inst_features, inst_loc = instance_mask_to_features(instance_masks[i], dense_features[i])
    instance_features= np.concatenate((instance_features, inst_features), axis=0)
    frames= np.concatenate((frames, np.ones(inst_features.shape[0])*i), axis=0)
    inst_locs= np.concatenate((inst_locs, inst_loc), axis=0)
    

# contrained DBSCAN tracking
labels = constrained_dbscan_tracking(instance_features, frames, eps=1.5, min_samples=2)

napari_tracks= np.concatenate([labels[:, np.newaxis],frames[:, np.newaxis], inst_locs.astype(np.int32)], axis=1).astype(np.int32)
napari_tracks_filtered=napari_tracks[napari_tracks[:,0]!=-1]
print(f"Found {len(np.unique(labels)) - 1} unique tracks (excluding noise).")
print(f"Track ID distribution: {np.unique(labels, return_counts=True)}")

viewer = napari.Viewer()
video_path="/home/tristan/data/CTC_SINETRA_seg/Fluo-N2DH-GOWT1/02/video.tiff"
viewer.add_image(tifffile.imread(video_path)[2:n_tested+2], name="video")
viewer.add_labels(instance_masks, name="instance masks")
viewer.add_tracks(napari_tracks_filtered)