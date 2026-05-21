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

# Sort by date
print(f"\nSorting by date...")
filtered_lineups['Date'] = pd.to_datetime(filtered_lineups['Date'])
filtered_lineups = filtered_lineups.sort_values('Date').reset_index(drop=True)

# Save filtered lineups
print(f"\nSaving filtered lineups to {LINEUPS_FILE}...")
filtered_lineups.to_csv(LINEUPS_FILE, index=False)
print(f"Successfully saved {filtered_count} rows")

print("\n" + "=" * 60)
print("Filtering completed successfully!")
print("=" * 60)
