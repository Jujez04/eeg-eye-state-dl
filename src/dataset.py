import json
from pathlib import Path
from urllib.request import urlopen, Request
import pandas as pd

EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

OPENML_URL = "https://www.openml.org/api/v1/json/data/1471"

def load_dataset_raw(dir: str | Path = "data/raw") -> pd.DataFrame:
    # Check if file already exist in data directory
    raw_path = Path(dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    cache_file = raw_path / "eeg_eye_state.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    # Fetch metadata from OpenML URL
    req = Request(OPENML_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    file_url = meta["data_set_description"]["url"]

    # Fetch raw dataset using the file url fetched before
    req_data = Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req_data) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    data_start = next(
        i + 1 for i, line in enumerate(lines)
        if line.strip().lower() == "@data"
    )

    # Read ARFF file
    rows = [
        line.strip().split(",")
        for line in lines[data_start:]
        if line.strip() and not line.strip().startswith("%")
    ]

    columns = EEG_CHANNELS + ["eye_closed"]
    df = pd.DataFrame(rows, columns=columns)
    for ch in EEG_CHANNELS:
        df[ch] = df[ch].astype(float)
    df["eye_closed"] = df["eye_closed"].str.strip("'\"").astype(int)

    # Cache file
    df.to_csv(cache_file, index=False)
    return df