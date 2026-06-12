#!/usr/bin/env python3
"""Test that a2 position classification now matches a1 exactly."""

from a2_extract_historic_lineups import get_position_category

# Test cases from Transfermarkt position labels
test_cases = [
    # Position, Expected Category, Priority Level
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
    ('Left Midfield', 'Midfielder', '3 - CRITICAL FIX'),
    ('Right Midfield', 'Midfielder', '3 - CRITICAL FIX'),
    ('Midfield', 'Midfielder', '3'),
    ('Centre-Back', 'Defender', '4'),
    ('Left-Back', 'Defender', '4'),
    ('Right-Back', 'Defender', '4'),
    ('Full Back', 'Defender', '4'),
    ('Wing-Back', 'Defender', '4'),
    ('Defender', 'Defender', '4'),
    ('Sweeper', 'Defender', '4'),
    ('Libero', 'Defender', '4'),
]

errors = []
for pos, expected, level in test_cases:
    result = get_position_category(pos)
    status = "✓" if result == expected else "✗"
    print(f"{status} {pos:25} -> {result:12} (expected {expected:12}) [{level}]")
    if result != expected:
        errors.append(f"{pos} returned {result} instead of {expected}")

print("\n" + "="*70)
if errors:
    print(f"FAILED: {len(errors)} test cases")
    for err in errors:
        print(f"  - {err}")
    exit(1)
else:
    print(f"✓ ALL {len(test_cases)} TEST CASES PASSED!")
    print("✓ a2 position logic now matches a1 exactly")
    exit(0)
