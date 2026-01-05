#!/usr/bin/env python3
"""
Fix duplicate problem entries in run_info.json.
Keeps the most recent entry for each problem_id.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def fix_run_info_duplicates(run_info_path: Path):
    """Remove duplicate problem entries, keeping the most recent."""

    # Load run_info.json
    with open(run_info_path, 'r', encoding='utf-8') as f:
        run_info = json.load(f)

    original_count = len(run_info.get('problems', []))
    print(f"Original problem count: {original_count}")

    # Create backup
    backup_path = run_info_path.parent / f"run_info_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(f"✓ Backup created: {backup_path}")

    # Remove duplicates (keep most recent by using dict - last one wins)
    problems_dict = {}
    for problem in run_info.get('problems', []):
        problem_id = problem.get('problem_id')
        if problem_id:
            # If problem_id already exists, compare timestamps and keep the later one
            if problem_id in problems_dict:
                existing = problems_dict[problem_id]
                existing_time = existing.get('start_time', '')
                new_time = problem.get('start_time', '')

                # Keep the one with later start_time
                if new_time > existing_time:
                    problems_dict[problem_id] = problem
            else:
                problems_dict[problem_id] = problem

    # Update run_info with deduplicated problems
    run_info['problems'] = list(problems_dict.values())

    # Sort by start_time for better readability
    run_info['problems'].sort(key=lambda p: p.get('start_time', ''))

    new_count = len(run_info['problems'])
    removed = original_count - new_count

    print(f"Deduplicated problem count: {new_count}")
    print(f"Removed {removed} duplicate entries")

    # Save fixed run_info.json
    with open(run_info_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"✓ Fixed run_info.json saved")

    # Show duplicate problem IDs that were removed
    problem_ids = [p.get('problem_id') for p in run_info.get('problems', [])]
    duplicates_found = original_count - new_count
    if duplicates_found > 0:
        print(f"\n⚠️  Found and removed {duplicates_found} duplicate entries")

    return removed

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_run_info_duplicates.py <path_to_run_info.json>")
        print("Example: python fix_run_info_duplicates.py agent_logs/gpt-5.1_no_vision_20260105_101316/run_info.json")
        sys.exit(1)

    run_info_path = Path(sys.argv[1])

    if not run_info_path.exists():
        print(f"Error: File not found: {run_info_path}")
        sys.exit(1)

    print(f"Fixing duplicates in: {run_info_path}\n")
    fix_run_info_duplicates(run_info_path)
    print("\n✓ Done!")

if __name__ == "__main__":
    main()
