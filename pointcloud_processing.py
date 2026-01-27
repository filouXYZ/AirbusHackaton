import numpy as np
import pandas as pd

def clean_lidar_points(df, max_distance_m=150.0):
    """
    Remove invalid lidar points and limit max range.
    """
    # distance_cm == 0 → no return
    df = df[df["distance_cm"] > 0]

    # limit range
    df = df[df["distance_cm"] <= max_distance_m * 100]

    return df.reset_index(drop=True)


def spherical_to_cartesian(df):
    """
    Convert spherical lidar coordinates to local Cartesian coordinates (meters).
    
    Returns:
        xyz: (N, 3) numpy array
    """
    distance_m = df["distance_cm"].to_numpy() / 100.0
    azimuth = np.deg2rad(df["azimuth_raw"].to_numpy() / 100.0)
    elevation = np.deg2rad(df["elevation_raw"].to_numpy() / 100.0)

    x = distance_m * np.cos(elevation) * np.cos(azimuth)
    y = -distance_m * np.cos(elevation) * np.sin(azimuth)
    z = distance_m * np.sin(elevation)

    xyz = np.stack((x, y, z), axis=1)
    return xyz


def extract_features(df):
    """
    Extract per-point features (excluding xyz).
    """
    features = {}

    if "reflectivity" in df.columns:
        features["reflectivity"] = df["reflectivity"].to_numpy()[:, None]

    if {"r", "g", "b"}.issubset(df.columns):
        features["rgb"] = df[["r", "g", "b"]].to_numpy()

    return features


def process_frame(frame):
    """
    Full processing pipeline for one frame.

    Input:
        frame = {
          "pose": (ego_x, ego_y, ego_z, ego_yaw),
          "points": NxD numpy array
        }

    Output:
        processed_frame = {
          "pose": pose,
          "xyz": (N,3),
          "features": dict,
          "raw_df": cleaned DataFrame
        }
    """
    columns = frame["points"].dtype.names
    df = pd.DataFrame(frame["points"], columns=columns)

    df = clean_lidar_points(df)

    xyz = spherical_to_cartesian(df)
    features = extract_features(df)

    return {
        "pose": frame["pose"],
        "xyz": xyz,
        "features": features,
        "raw_df": df
    }
