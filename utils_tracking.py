import numpy as np
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import byotrack
import torch
from byotrack.api.detector.detections import Detections
#regionprops
from skimage.measure import regionprops
def _process_frame(args):
    t, mask, avg_feat_dict, var_dict = args

    ids = []
    time = []
    features = []
    variance = []
    positions = []

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background
    props = regionprops(mask)
    for uid in unique_ids:
        ids.append(uid)
        time.append(t)
        features.append(avg_feat_dict[uid])
        variance.append(var_dict[uid])
        positions.append(props[uid-1].centroid)
    return ids, time, features, variance, positions


def reformat_model_output(instance_masks, per_instance_features, per_instance_variance):
    """
    Reformats the model output to match the expected input for the linker.
    """
    tasks = [
        (t, mask, avg_feat_dict, var_dict)
        for t, (mask, avg_feat_dict, var_dict) in enumerate(
            zip(instance_masks, per_instance_features, per_instance_variance)
        )
    ]

    ids = []
    time = []
    features = []
    variance = []
    positions = []

    with ProcessPoolExecutor() as executor:
        for frame_ids, frame_time, frame_features, frame_variance, frame_positions in tqdm(
            executor.map(_process_frame, tasks), total=len(tasks)
        ):
            ids.extend(frame_ids)
            time.extend(frame_time)
            features.extend(frame_features)
            variance.extend(frame_variance)
            positions.extend(frame_positions)

    return ids, time, features, variance, positions




def reformat_model_output_as_byotrack_detections_single_thread(instance_masks, per_instance_features, per_instance_variance):
    """
    Reformats the model output to match the expected input for the linker.
    """
    detections=[]
    for frame in tqdm(range(len(instance_masks))):
        ids,time,features,variance,positions=_process_frame((frame, instance_masks[frame], per_instance_features[frame], per_instance_variance[frame]))
        coords_torch=torch.from_numpy(np.array(positions)).float()
        embeddings_torch=torch.from_numpy(np.array(features)).float()
        mask_dict={"position":coords_torch,"features":embeddings_torch}
        detections.append(byotrack.api.detector.detections.Detections(mask_dict))


    return detections


def reformat_model_output_as_byotrack_detections(instance_masks, per_instance_features, per_instance_variance):
    """
    Reformats the model output to match the expected input for the linker.
    """
    tasks = [
        (frame, instance_masks[frame], per_instance_features[frame], per_instance_variance[frame])
        for frame in range(len(instance_masks))
    ]

    detections = []
    with ProcessPoolExecutor() as executor:
        for _, _, features, _, positions in tqdm(
            executor.map(_process_frame, tasks), total=len(tasks)
        ):
            coords_torch = torch.from_numpy(np.asarray(positions)).float()
            embeddings_torch = torch.from_numpy(np.asarray(features)).float()
            mask_dict = {"position": coords_torch, "features": embeddings_torch}
            detections.append(Detections(mask_dict))

    return detections





def _tracks_to_napari(tracks: Sequence[byotrack.Track]) -> np.ndarray:
    """ Convert byotrack tracks to a format compatible with napari (T, D, H, W)"""
    napari_tracks = []
    for i,track in enumerate(tracks):
        for n_alive in range(len(track)):

            if track.points[n_alive][0].isnan():
                # The track is not alive at this time point, we skip it
                continue

            napari_tracks.append([i, track.start + n_alive, *track.points[n_alive].cpu().numpy()])
    return np.array(napari_tracks)
