import soccerdata as sd
from pathlib import Path

base_dir = Path.cwd() / "fbref_data" 

fbref = sd.FBref(
    leagues="GER-Bundesliga",
    seasons=[2024],
    data_dir=Path.cwd() / "data" / "fbref"
)

stats = fbref.read_team_match_stats()

print(stats.head())

stats.to_csv(base_dir / "stats.csv", index=False)