#!/usr/bin/env python3
"""
Fix verification_conditions format issues in geoqa3_dataset.json

Issues to fix:
1. point_on_circle: Convert 'segment' parameter to 'circle_center'
2. tangent: Convert 'segment' parameter to 'line'
3. (Add more fixes as needed)
"""

import json
import shutil
from datetime import datetime
from collections import defaultdict


def backup_file(filepath):
    """Create a backup of the original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def fix_point_on_circle(condition):
    """
    Fix point_on_circle conditions with 'segment' parameter.

    Convert: {'point': 'B', 'segment': ['O', 'B']}
    To:      {'point': 'B', 'circle_center': 'O'}
    """
    if 'segment' in condition and 'circle_center' not in condition:
        segment = condition['segment']
        if isinstance(segment, list) and len(segment) == 2:
            # Assume first element is circle center
            circle_center = segment[0]
            condition['circle_center'] = circle_center
            del condition['segment']
            return True, f"Converted segment {segment} to circle_center '{circle_center}'"
    return False, None


def fix_tangent(condition):
    """
    Fix tangent conditions with 'segment' parameter.

    Convert: {'circle_center': 'I', 'point': 'D', 'segment': ['B', 'C']}
    To:      {'circle_center': 'I', 'point': 'D', 'line': ['B', 'C']}

    Note: 'segment' should be 'line' for tangent validation
    """
    if 'segment' in condition and 'line' not in condition:
        segment = condition['segment']
        condition['line'] = segment
        del condition['segment']
        return True, f"Converted segment to line: {segment}"
    return False, None


def fix_circle_radius_key(circle_def):
    """
    Fix circle definitions with wrong 'radius' key.

    Convert: {'center': 'O', 'radius': 6}
    To:      {'center': 'O', 'radius_length': 6}
    """
    if 'radius' in circle_def and 'radius_length' not in circle_def and 'radius_point' not in circle_def:
        radius_value = circle_def['radius']
        circle_def['radius_length'] = radius_value
        del circle_def['radius']
        return True, f"Converted 'radius' to 'radius_length': {radius_value}"
    return False, None


def analyze_and_fix_dataset(input_path, output_path=None, dry_run=False):
    """
    Analyze and fix verification_conditions in the dataset.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, overwrites input)
        dry_run: If True, only report issues without making changes
    """
    print("="*80)
    print("DATASET VERIFICATION CONDITIONS FIXER")
    print("="*80)

    # Load dataset
    print(f"\n📖 Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_problems = len(data.get('problems', []))
    print(f"   Total problems: {total_problems}")

    # Track fixes
    fix_stats = defaultdict(lambda: {'count': 0, 'examples': []})
    problems_modified = 0

    # Process each problem
    for problem_idx, problem in enumerate(data['problems']):
        problem_id = problem.get('id', f'index_{problem_idx}')
        problem_modified = False

        if 'verification_conditions' not in problem:
            continue

        for cond_idx, condition in enumerate(problem['verification_conditions']):
            cond_type = condition.get('type', 'UNKNOWN')

            # Apply fixes based on condition type
            fixed = False
            fix_msg = None

            if cond_type == 'point_on_circle':
                fixed, fix_msg = fix_point_on_circle(condition)
                if fixed:
                    fix_stats['point_on_circle']['count'] += 1
                    fix_stats['point_on_circle']['examples'].append({
                        'problem_id': problem_id,
                        'condition_index': cond_idx,
                        'fix': fix_msg,
                        'result': dict(condition)
                    })
                    problem_modified = True

            elif cond_type == 'tangent':
                fixed, fix_msg = fix_tangent(condition)
                if fixed:
                    fix_stats['tangent']['count'] += 1
                    fix_stats['tangent']['examples'].append({
                        'problem_id': problem_id,
                        'condition_index': cond_idx,
                        'fix': fix_msg,
                        'result': dict(condition)
                    })
                    problem_modified = True

        # Fix required_objects.circles
        if 'required_objects' in problem and 'circles' in problem['required_objects']:
            for circle_idx, circle_def in enumerate(problem['required_objects']['circles']):
                fixed, fix_msg = fix_circle_radius_key(circle_def)
                if fixed:
                    fix_stats['circle_radius_key']['count'] += 1
                    fix_stats['circle_radius_key']['examples'].append({
                        'problem_id': problem_id,
                        'circle_index': circle_idx,
                        'fix': fix_msg,
                        'result': dict(circle_def)
                    })
                    problem_modified = True

        if problem_modified:
            problems_modified += 1

    # Print summary
    print("\n" + "="*80)
    print("FIX SUMMARY")
    print("="*80)

    if not fix_stats:
        print("\n✓ No issues found! Dataset is already in correct format.")
    else:
        print(f"\n📊 Found and {'would fix' if dry_run else 'fixed'} issues in {problems_modified} problems:")

        for cond_type, stats in sorted(fix_stats.items()):
            print(f"\n{cond_type}:")
            print(f"  - Total fixes: {stats['count']}")
            print(f"  - Examples (showing first 3):")
            for i, example in enumerate(stats['examples'][:3], 1):
                # Handle both condition_index and circle_index
                if 'condition_index' in example:
                    location = f"condition #{example['condition_index']}"
                elif 'circle_index' in example:
                    location = f"circle #{example['circle_index']}"
                else:
                    location = "unknown location"
                print(f"    {i}. Problem {example['problem_id']}, {location}")
                print(f"       {example['fix']}")
                print(f"       Result: {example['result']}")

    # Save fixed dataset
    if not dry_run and fix_stats:
        # Create backup
        backup_path = backup_file(input_path)

        # Determine output path
        if output_path is None:
            output_path = input_path

        # Update metadata
        if 'metadata' in data:
            data['metadata']['last_fixed'] = datetime.now().isoformat()
            data['metadata']['fixes_applied'] = {
                cond_type: stats['count']
                for cond_type, stats in fix_stats.items()
            }

        # Save
        print(f"\n💾 Saving fixed dataset to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Dataset fixed and saved!")

    elif dry_run:
        print(f"\n🔍 DRY RUN - No changes were made to the dataset")

    print("\n" + "="*80)
    return fix_stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix verification_conditions format issues in geoqa3_dataset.json'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='data/geoqa3_dataset.json',
        help='Input JSON file (default: data/geoqa3_dataset.json)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (default: overwrite input file)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )

    args = parser.parse_args()

    analyze_and_fix_dataset(
        args.input_file,
        output_path=args.output,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
