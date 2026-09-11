#!/usr/bin/env python3
import pandas as pd, os, glob, re, unicodedata

# 1. Load the predictions CSV
pred = pd.read_csv('5_new_prediction_dashboard/weekly_results/weekly_predictions_latest.csv')
csv_teams = sorted(set(pred['home_team']) | set(pred['away_team']))
print(f'=== {len(csv_teams)} teams in predictions CSV ===')
for t in csv_teams:
    print(t)

# 2. Load all logo filenames (basename without .png)
logos = []
for p in glob.glob('misc/logos/*/*.png'):
    league_dir = os.path.basename(os.path.dirname(p))
    name = os.path.splitext(os.path.basename(p))[0]
    logos.append((league_dir, name))
print(f'\n=== {len(logos)} logo files ===')

# 3. Normalized match
STOPWORDS = r'\b(fc|afc|cf|sc|sv|vfb|vfl|ac|as|ss|ssc|us|rc|ogc|losc|gd|cd|cs|sl|cp|fk|if|bk|ff|sk|kv|kaa|krc|rsc|bsc|tsg|rb|club|calcio|de|futbol|football|the)\b'

def norm(s):
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(STOPWORDS, ' ', s)
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

logo_by_norm = {}
for league_dir, name in logos:
    logo_by_norm.setdefault(norm(name), (league_dir, name))

print('\n=== Match report ===')
unmatched = []
for t in csv_teams:
    key = norm(t)
    if key in logo_by_norm:
        ld, ln = logo_by_norm[key]
        if ln.lower() != t.lower():
            print(f'  FUZZY: {t!r:40} -> {ln!r}  [{ld}]')
    else:
        hits = [(ld, ln) for k, (ld, ln) in logo_by_norm.items() if key and (key in k or k in key)]
        if hits:
            print(f'  SUBSTR: {t!r:40} -> {hits[:3]}')
        else:
            unmatched.append(t)

print(f'\n=== {len(unmatched)} UNMATCHED teams ===')
for t in unmatched:
    print(f'  {t}')