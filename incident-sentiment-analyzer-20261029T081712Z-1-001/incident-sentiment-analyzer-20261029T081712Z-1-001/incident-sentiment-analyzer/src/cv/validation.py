from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def split_video_dataset(
    df_videos: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Stratified two-step split at video level
    stratify_col = df_videos["label"] if "label" in df_videos.columns else None
    
    df_train, df_temp = train_test_split(
        df_videos,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_col
    )
    
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_temp = df_temp["label"] if "label" in df_temp.columns else None
    
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - relative_val_ratio),
        random_state=random_state,
        stratify=stratify_temp
    )
    
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    
    return df_train, df_val, df_test

def run_leakage_and_integrity_check(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    solution_csv: str | Path
) -> bool:
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    test_csv = Path(test_csv)
    solution_csv = Path(solution_csv)

    df_train = pd.read_csv(train_csv, dtype={"video_id": str})
    df_val = pd.read_csv(val_csv, dtype={"video_id": str})
    df_test = pd.read_csv(test_csv, dtype={"video_id": str})
    df_sol = pd.read_csv(solution_csv, dtype={"id": str})

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test), ("solution", df_sol)]:
        assert df.isna().sum().sum() == 0, f"NaN values found in {name} CSV!"

    train_ids = set(df_train["video_id"].astype(str).str.zfill(5))
    val_ids = set(df_val["video_id"].astype(str).str.zfill(5))
    test_ids = set(df_test["video_id"].astype(str).str.zfill(5))

    assert len(train_ids & val_ids) == 0, "Leakage: Train & Val overlap!"
    assert len(train_ids & test_ids) == 0, "Leakage: Train & Test overlap!"
    assert len(val_ids & test_ids) == 0, "Leakage: Val & Test overlap!"

    for df in [df_train, df_val, df_test]:
        counts = df.groupby("video_id").size()
        assert (counts >= 5).all(), "Some videos have fewer than 5 frames!"

    sol_ids = df_sol["id"].astype(str).str.zfill(5)
    assert (sol_ids.str.len() == 5).all(), "Solution IDs must be 5-digit zero-padded strings!"
    assert set(sol_ids) == test_ids, "Solution IDs do not match test video IDs!"

    for p in df_train["frame_path"].sample(min(5, len(df_train)), random_state=42):
        resolved_p = Path(p)
        if not resolved_p.is_absolute():
            resolved_p = train_csv.resolve().parents[2] / resolved_p
        assert resolved_p.exists() or Path(p).exists(), f"Sample frame path does not exist on disk: {p}"

    print("All integrity and sanity checks passed successfully.")
    return True