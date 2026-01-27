import h5py
import numpy as np
import pandas as pd

POSE_FIELDS = ["ego_x", "ego_y", "ego_z", "ego_yaw"]

def load_h5_data(file_path, dataset_name="lidar_points"):
    """
    Load an HDF5 lidar file into a Pandas DataFrame.
    Each row corresponds to one lidar point.
    """
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")
        data = f[dataset_name][:]

    # Structured array → DataFrame
    df = pd.DataFrame({name: data[name] for name in data.dtype.names})
    return df


def get_unique_poses(df):
    """
    Identify unique frames using the ego pose quadruplet.
    Returns a DataFrame with a pose_index.
    """
    if not all(field in df.columns for field in POSE_FIELDS):
        raise ValueError("Pose fields not found in dataframe")

    pose_df = (
        df.groupby(POSE_FIELDS)
        .size()
        .reset_index(name="num_points")
        .reset_index(names="pose_index")
    )
    return pose_df


def filter_by_pose(df, pose_row):
    """
    Filter lidar points corresponding to one frame (one ego pose).
    """
    mask = np.ones(len(df), dtype=bool)
    for field in POSE_FIELDS:
        mask &= df[field] == pose_row[field]

    return df[mask].reset_index(drop=True)


def build_frames(df):
    """
    Build a list of frames from a dataframe.
    Each frame contains:
      - pose: (ego_x, ego_y, ego_z, ego_yaw)
      - points: numpy array (N, D)
    """
    frames = []
    pose_table = get_unique_poses(df)

    for _, pose_row in pose_table.iterrows():
        frame_df = filter_by_pose(df, pose_row)

        pose = tuple(pose_row[field] for field in POSE_FIELDS)
        points = frame_df.to_numpy()  # NxD (all point features)

        frames.append({
            "pose": pose,
            "points": points
        })

    return frames
