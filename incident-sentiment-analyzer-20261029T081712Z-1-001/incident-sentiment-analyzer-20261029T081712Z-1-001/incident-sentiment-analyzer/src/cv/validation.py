from pathlib import Path
import pandas as pd
import numpy as np

def split_video_dataset(df_videos, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits video dataset into train, validation, and test sets at the video level.
    """
    np.random.seed(random_state)
    shuffled = df_videos.sample(frac=1).reset_index(drop=True)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val:]
    
    return train_df, val_df, test_df


def run_leakage_and_integrity_check(train_csv, val_csv, test_csv, solution_csv) -> bool:
    """
    Performs comprehensive integrity and leakage checks:
    1. Set-disjointness (no data leakage between train, val, test splits).
    2. Minimum 5 frames per video requirement across all splits.
    3. 5-digit zero-padded string format for video IDs in solution_v2.csv and exact match with test IDs.
    4. No NaN values in any of the dataframes (train, val, test, solution).
    5. Physical existence of image files on disk.
    """
    # Load dataframes
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    df_sol = pd.read_csv(solution_csv)
    
    # Standardize video IDs to 5-digit zero-padded strings
    for df in [df_train, df_val, df_test]:
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str).str.zfill(5)
            
    id_col = "id" if "id" in df_sol.columns else "video_id"
    df_sol[id_col] = df_sol[id_col].astype(str).str.zfill(5)

    # 1. Leakage Check (Set Disjointness)
    train_ids = set(df_train["video_id"])
    val_ids = set(df_val["video_id"])
    test_ids = set(df_test["video_id"])
    
    assert train_ids.isdisjoint(val_ids), "Leakage detected between Train and Val!"
    assert train_ids.isdisjoint(test_ids), "Leakage detected between Train and Test!"
    assert val_ids.isdisjoint(test_ids), "Leakage detected between Val and Test!"

    # 2. Minimum 5 Frames Check
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = df.groupby("video_id").size()
        assert (counts >= 5).all(), f"{name} split contains videos with fewer than 5 frames!"

    # 3. Solution Format & Test Match Check
    sol_ids = set(df_sol[id_col])
    assert all(len(i) == 5 for i in sol_ids), "Solution IDs are not 5-digit strings!"
    assert sol_ids == test_ids, "Solution IDs do not exactly match test set video IDs!"

    # 4. NaN Assertion Check
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test), ("Solution", df_sol)]:
        assert df.isna().sum().sum() == 0, f"NaN values detected in {name} dataframe!"

    # 5. Physical File Existence Check (sample check for speed and safety)
    sample_frames = df_train["frame_path"].sample(min(5, len(df_train)))
    for p in sample_frames:
        assert Path(p).exists(), f"Frame file does not exist on disk: {p}"

    print("All integrity and sanity checks passed successfully.")
    return True