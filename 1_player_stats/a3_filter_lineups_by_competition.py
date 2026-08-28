"""
Filter lineups.csv to keep only the top 5 European competitions
"""

import pandas as pd
from pathlib import Path

# Define paths
DATA_DIR = Path('data')
LINEUPS_FILE = DATA_DIR / 'lineups.csv'
LINEUPS_BACKUP_FILE = DATA_DIR / 'lineups_backup_before_filter.csv'

# Define target competitions (consistent with existing format)
TARGET_COMPETITIONS = [
    'serie-a',
    'premier-league',
    'bundesliga',
    'laliga',
    'ligue-1'
]

print("=" * 60)
print("Filtering Lineups to Top 5 European Competitions")
print("=" * 60)

# Load lineups
print(f"\nLoading {LINEUPS_FILE}...")
lineups = pd.read_csv(LINEUPS_FILE, low_memory=False)
original_count = len(lineups)
print(f"Original rows: {original_count}")

# Check competitions before filtering
print(f"\nCompetitions before filtering:")
comp_counts = lineups['Competition'].value_counts()
print(comp_counts)

# Create backup before filtering
print(f"\nCreating backup...")
lineups.to_csv(LINEUPS_BACKUP_FILE, index=False)
print(f"Backup saved to {LINEUPS_BACKUP_FILE}")

# Filter to target competitions
print(f"\nFiltering to target competitions: {', '.join(TARGET_COMPETITIONS)}")
filtered_lineups = lineups[lineups['Competition'].isin(TARGET_COMPETITIONS)].copy()
filtered_count = len(filtered_lineups)

print(f"\nRows after filtering: {filtered_count}")
print(f"Rows removed: {original_count - filtered_count}")

# Check competitions after filtering
print(f"\nCompetitions after filtering:")
filtered_comp_counts = filtered_lineups['Competition'].value_counts()
print(filtered_comp_counts)

# Verify all target competitions are present
missing_comps = set(TARGET_COMPETITIONS) - set(filtered_lineups['Competition'].unique())
if missing_comps:
    print(f"\nWarning: Missing competitions: {missing_comps}")
else:
    print(f"\n✓ All target competitions present")

# filter to keep only rows where team names consist of three uppercase letters (e.g., "MUN", "BAR", "PSG") in either Home Team or Away Team columns
print(f"\nFiltering to rows with three-letter uppercase team codes...")
filtered_lineups = filtered_lineups[
    filtered_lineups['Home Team'].str.match(r'^[A-Z]{3}$') |
    filtered_lineups['Away Team'].str.match(r'^[A-Z]{3}$')
].copy()
filtered_count = len(filtered_lineups)
print(f"Rows after team code filtering: {filtered_count}")

# Sort by date
print(f"\nSorting by date...")
filtered_lineups['Date'] = pd.to_datetime(filtered_lineups['Date'])
filtered_lineups = filtered_lineups.sort_values('Date').reset_index(drop=True)

# add a game_id column by concatenating date, home team, and away team (e.g., "20230812_MUN_BAR")
# the id column should be the first column
print(f"\nAdding game_id column...")
filtered_lineups['game_id'] = filtered_lineups['Date'].dt.strftime('%Y%m%d') + '_' + filtered_lineups['Home Team'] + '_' + filtered_lineups['Away Team']
cols = filtered_lineups.columns.tolist()
cols = ['game_id'] + [col for col in cols if col != 'game_id']
filtered_lineups = filtered_lineups[cols]   

# ensure that the game_id column is unique and remove any duplicate rows based on game_id
print(f"\nRemoving duplicate game_id rows...")
filtered_lineups = filtered_lineups.drop_duplicates(subset=['game_id'], keep='first')
filtered_count = len(filtered_lineups)
print(f"Rows after removing duplicates: {filtered_count}")

# Save filtered lineups
print(f"\nSaving filtered lineups to {LINEUPS_FILE}...")
filtered_lineups.to_csv(LINEUPS_FILE, index=False)
print(f"Successfully saved {filtered_count} rows")

print("\n" + "=" * 60)
print("Filtering completed successfully!")
print("=" * 60)
