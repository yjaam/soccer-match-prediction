#!/usr/bin/env python3
"""Test the corrected position function"""

import re
import pandas as pd

def get_position_category_FIXED(position):
    """Map Transfermarkt positions using the exact same logic as a1_scrape_new_formations.py
    
    a1 classification order:
    1. GK -> Goalkeeper
    2. Starts with "F" OR in ["ST", "RW", "LW", "A"] -> Attacker
    3. Starts with "M" OR starts with "AM" OR starts with "DM" -> Midfielder
    4. Starts with "D" AND NOT starts with "DM" -> Defender
    5. Else -> Midfielder
    
    KEY FIX: Check exact winger matches BEFORE checking for "wing-" prefix to avoid catching "wing-back"
    """
    position = re.sub(r"[\s\-_]+", " ", str(position).lower()).strip() if pd.notna(position) else ""

    if not position:
        return 'Midfielder'

    # 1. GK -> Goalkeeper
    if position in {'gk', 'goalkeeper', 'keeper'}:
        return 'Goalkeeper'

    # 2. Attacker: Check exact matches first (to handle winger without catching wing-back)
    # Then check prefixes
    attacker_exact = {'left winger', 'right winger', 'winger', 'attacker', 'a', 'st', 'rw', 'lw'}
    if position in attacker_exact:
        return 'Attacker'
    
    # Check for forward/striker prefixes (need "f " with space to avoid matching "full" or "wing")
    # Actually, check exact prefixes more carefully
    if position.startswith('centre forward') or position.startswith('center forward') or position.startswith('second striker'):
        return 'Attacker'
    
    # Check if starts with "f" and next char is space or hyphen (to catch "Forward", "F " etc)
    if position in {item for prefix in ['f ', 'forward', 'forward '] for item in [prefix]}:
        return 'Attacker'
    
    # Simple check: if it starts with just "f" and nothing else complex, it's forward
    if position.startswith('f') and not position.startswith('f back') and not position.startswith('full'):
        return 'Attacker'

    if position.startswith('str'):  # striker
        return 'Attacker'

    # 3. Midfielder: Check for midfield positions
    if position in {'dm', 'cm', 'am', 'm'}:
        return 'Midfielder'
    
    if any(position.startswith(prefix) for prefix in ['midfield', 'central midfield', 'centre midfield', 
                                                        'defensive midfield', 'attacking midfield', 
                                                        'left midfield', 'right midfield']):
        return 'Midfielder'

    # 4. Defender: Check for defense positions
    defender_exact = {'cb', 'lb', 'rb', 'd'}
    if position in defender_exact:
        return 'Defender'
    
    defender_prefixes = {'centre back', 'center back', 'left back', 'right back', 'full back', 'wing back', 
                         'defender', 'sweeper', 'libero'}
    if any(position.startswith(prefix) for prefix in defender_prefixes) and not position.startswith('dm'):
        return 'Defender'

    # 5. Default else -> Midfielder
    return 'Midfielder'


# Test cases
test_cases = [
    ('Goalkeeper', 'Goalkeeper', '1'),
    ('Centre-Forward', 'Attacker', '2'),
    ('Second Striker', 'Attacker', '2'),
    ('Striker', 'Attacker', '2'),
    ('Left Winger', 'Attacker', '2'),
    ('Right Winger', 'Attacker', '2'),
    ('Winger', 'Attacker', '2'),
    ('Attacker', 'Attacker', '2'),
    ('Centre Midfield', 'Midfielder', '3'),
    ('Defensive Midfield', 'Midfielder', '3'),
    ('Attacking Midfield', 'Midfielder', '3'),
    ('Left Midfield', 'Midfielder', '3 - CRITICAL'),
    ('Right Midfield', 'Midfielder', '3 - CRITICAL'),
    ('Midfield', 'Midfielder', '3'),
    ('Centre-Back', 'Defender', '4'),
    ('Left-Back', 'Defender', '4'),
    ('Right-Back', 'Defender', '4'),
    ('Full Back', 'Defender', '4 - BUG FIX'),
    ('Wing-Back', 'Defender', '4 - BUG FIX'),
    ('Defender', 'Defender', '4'),
    ('Sweeper', 'Defender', '4'),
    ('Libero', 'Defender', '4'),
]

errors = []
for pos, expected, level in test_cases:
    result = get_position_category_FIXED(pos)
    status = "✓" if result == expected else "✗"
    print(f"{status} {pos:25} -> {result:12} (expected {expected:12}) [{level}]")
    if result != expected:
        errors.append(f"{pos} returned {result} instead of {expected}")

print("\n" + "="*70)
if errors:
    print(f"FAILED: {len(errors)} test cases")
    for err in errors:
        print(f"  - {err}")
else:
    print(f"✓ ALL {len(test_cases)} TEST CASES PASSED!")
