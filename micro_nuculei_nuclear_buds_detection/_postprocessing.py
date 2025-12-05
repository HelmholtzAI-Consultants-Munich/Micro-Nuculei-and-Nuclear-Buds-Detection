# Postprocessing module for the Micro-Nuculei and Nuclear Buds Detection plugin.

# each detection is assigned to a segmented nuclei based on the closest distance to the nucleus contur

from ._constants import CLASS_IDS
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from ._constants import POSTPROCESSING_SUBFOLDER

def postprocess_detections(dataset_path: Path, image_path: Path, annotation_file: Path, nuclei_segmentation_file: Path):
    """
    Postprocess the detections based on the nuclei segmentation.
    """
    
    detections = np.loadtxt(annotation_file, delimiter=' ', dtype=np.float32, ndmin=2)
    if detections.size == 0:
        return
    nuclei_segmentation = np.load(nuclei_segmentation_file)
    if nuclei_segmentation.size == 0:
        return
    labels = np.unique(nuclei_segmentation)
    labels = labels[labels > 0]
    print("labels: ", len(labels))
    H, W = nuclei_segmentation.shape

    # revert detections normalization
    detections[:, 1] = detections[:, 1] * W
    detections[:, 2] = detections[:, 2] * H
    detections[:, 3] = detections[:, 3] * W
    detections[:, 4] = detections[:, 4] * H

    # get the contour points and labels of the nuclei
    contour_points, labs = find_contour_points(nuclei_segmentation)
    if contour_points.size == 0:  # Handle case where no contours are found
        return
    kd_tree = KDTree(contour_points)

    # assign the detections to the nuclei
    micro_nuclei_detections = detections[detections[:, 0] == CLASS_IDS["micro_nuclei"]]
    if micro_nuclei_detections.size > 0:
        _, micro_nuclei_idx = kd_tree.query(micro_nuclei_detections[:, [2, 1]])
        # Handle both scalar and array returns from kd_tree.query
        if np.isscalar(micro_nuclei_idx):
            micro_nuclei_labels = [labs[micro_nuclei_idx].item()]
        else:
            micro_nuclei_labels = [labs[idx].item() for idx in micro_nuclei_idx]
    else:
        micro_nuclei_labels = []
    print("micro_nuclei_labels: ", micro_nuclei_labels)

    nuclear_buds_detections = detections[detections[:, 0] == CLASS_IDS["nuclear_buds"]]
    if nuclear_buds_detections.size > 0:
        _, nuclear_buds_idx = kd_tree.query(nuclear_buds_detections[:, [1, 2]])
        # Handle both scalar and array returns from kd_tree.query
        if np.isscalar(nuclear_buds_idx):
            nuclear_buds_labels = [labs[nuclear_buds_idx].item()]
        else:
            nuclear_buds_labels = [labs[idx].item() for idx in nuclear_buds_idx]
    else:
        nuclear_buds_labels = []
    print("nuclear_buds_labels: ", nuclear_buds_labels)

    postprocessed_detections = pd.DataFrame(columns=["nucleus_id", "micro_nuclei", "nuclear_buds"])
    for lab in labels:
        micro_nuclei_count = micro_nuclei_labels.count(lab)
        nuclear_buds_count = nuclear_buds_labels.count(lab)
        postprocessed_detections.loc[lab] = [lab, micro_nuclei_count, nuclear_buds_count]
    # write postprocessed detections to a csv file
    try:
        relative_path = image_path.relative_to(dataset_path)
        # Create annotation path preserving subdirectory structure
        postprocessing_file = dataset_path / POSTPROCESSING_SUBFOLDER / relative_path.with_suffix('.csv')
        # Create parent directories if they don't exist
        postprocessing_file.parent.mkdir(parents=True, exist_ok=True)
    except ValueError:
        # Fallback: if path is not relative, use just the stem
        postprocessing_file = dataset_path / POSTPROCESSING_SUBFOLDER / f"{image_path.stem}.csv"
    postprocessed_detections.to_csv(postprocessing_file, index=False, sep='\t')

    # write summary statistics to a json file
    write_summary_statistics(dataset_path, image_path, postprocessed_detections)
    return


def find_contour_points(L):
    """
    Find the contour points of the nuclei. A point belongs the by shifting the nuclei_segmetnation matrix by 1 pixel in all directions
    the entry changes.
    """

    up = np.zeros_like(L); up[1:, :] = L[:-1, :]
    down = np.zeros_like(L); down[:-1, :] = L[1:, :]
    left = np.zeros_like(L); left[:, 1:] = L[:, :-1]
    right = np.zeros_like(L); right[:, :-1] = L[:, 1:]

    is_boundary = (
        (L != 0) & ( (up != L) | (down != L) | (left != L) | (right != L) )
    )

    xs, ys = np.nonzero(is_boundary)
    if xs.size == 0:
        return np.empty((0,2), dtype=np.int32), np.empty((0,), dtype=L.dtype)

    labs = L[xs, ys]
    coords = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    return coords, labs

def write_summary_statistics(dataset_path: Path, image_path: Path, postprocessed_detections: pd.DataFrame):
    """
    Write the summary statistics to a csv file.
    """
    postprocessing_file = dataset_path / POSTPROCESSING_SUBFOLDER / "summary_statistics.csv"
    postprocessing_file.parent.mkdir(parents=True, exist_ok=True)
    if postprocessing_file.exists():
        df = pd.read_csv(postprocessing_file, sep='\t')
    else:
        df = pd.DataFrame(columns=[
            "image_id",
            "micro_nuclei_count",
            "nuclear_buds_count",
            "nuclei_count",
            "1_micro_nuclei",
            "1_nuclear_buds",
            "2_micro_nuclei",
            "2_nuclear_buds",
            "3_micro_nuclei",
            "3_nuclear_buds",
            "4+_micro_nuclei",
            "4+_nuclear_buds",
        ])
    image_id = str(image_path.relative_to(dataset_path))
    # overwrite the row if the image_id already exists
    if image_id in df["image_id"].values:
        index = df[df["image_id"] == image_id].index[0]
    else:
        index = len(df)
    df.loc[index] = {
        "image_id": image_path.relative_to(dataset_path),
        "micro_nuclei_count": postprocessed_detections["micro_nuclei"].sum(),
        "nuclear_buds_count": postprocessed_detections["nuclear_buds"].sum(),
        "nuclei_count": postprocessed_detections["nucleus_id"].nunique(),
        "1_micro_nuclei": postprocessed_detections[postprocessed_detections["micro_nuclei"] == 1].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "1_nuclear_buds": postprocessed_detections[postprocessed_detections["nuclear_buds"] == 1].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "2_micro_nuclei": postprocessed_detections[postprocessed_detections["micro_nuclei"] == 2].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "2_nuclear_buds": postprocessed_detections[postprocessed_detections["nuclear_buds"] == 2].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "3_micro_nuclei": postprocessed_detections[postprocessed_detections["micro_nuclei"] == 3].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "3_nuclear_buds": postprocessed_detections[postprocessed_detections["nuclear_buds"] == 3].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "4+_micro_nuclei": postprocessed_detections[postprocessed_detections["micro_nuclei"] >= 4].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
        "4+_nuclear_buds": postprocessed_detections[postprocessed_detections["nuclear_buds"] >= 4].shape[0]/postprocessed_detections["nucleus_id"].nunique(),
    }

    df.to_csv(postprocessing_file, index=False, sep='\t')

