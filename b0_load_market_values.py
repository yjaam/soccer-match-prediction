from pathlib import Path
import shutil

import kagglehub

script_dir = Path(__file__).resolve().parent
transfermarkt_data_dir = script_dir.parent / "transfermarkt_data"

# Download latest version into the existing transfermarkt_data folder
download_path = Path(kagglehub.dataset_download("davidcariboo/player-scores"))
transfermarkt_data_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(download_path, transfermarkt_data_dir, dirs_exist_ok=True)

print("Path to dataset files:", transfermarkt_data_dir)