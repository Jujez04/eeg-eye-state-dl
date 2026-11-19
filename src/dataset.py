from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

EEG_EYE_STATE_CSV = "eeg_eye_state.csv"
DATA_ID = 1471
EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

def load_dataset_raw(dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the raw dataset from OpenML and save it to a CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Create the directory if it doesn't exist
    raw_path = Path(dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    cache_file = raw_path / EEG_EYE_STATE_CSV

    if cache_file.exists():
        return pd.read_csv(cache_file)
    
    # Fetch the dataset from OpenML
    eeg = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto")
    df = eeg.frame.copy()
    
    # Mapping between  V1,..,V14 with the real name of data features
    mapping = {f"V{i+1}": ch for i, ch in enumerate(EEG_CHANNELS)}
    df = df.rename(columns=mapping)
    df["eye_closed"] = df["Class"].astype(int)
    df = df.drop(columns=["Class"])
    return df