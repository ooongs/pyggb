#!/usr/bin/env python3
"""
Analyze GeoQA3 Dataset Statistics
Shows distribution by difficulty and category.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_dataset(dataset_path: str):
    """Analyze and print statistics from the dataset."""

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    problems = data.get('problems', [])
    metadata = data.get('metadata', {})

    print("="*80)
    print("GeoQA3 Dataset Statistics")
    print("="*80)

    # Metadata
    print("\n📊 Overall Metadata:")
    print(f"  Total problems in dataset: {metadata.get('total_problems', 'N/A')}")
    print(f"  Successfully parsed: {len(problems)}")
    print(f"  Skipped: {metadata.get('skipped', 'N/A')}")
    print(f"  Errors: {metadata.get('errors', 'N/A')}")
    print(f"  Created at: {metadata.get('created_at', 'N/A')}")

    # Initialize counters
    difficulty_counts = defaultdict(int)
    category_counts = defaultdict(int)
    difficulty_by_category = defaultdict(lambda: defaultdict(int))

    # Count statistics
    for problem in problems:
        difficulty = problem.get('difficulty', 'Unknown')
        category = problem.get('category', 'Unknown')

        difficulty_counts[difficulty] += 1
        category_counts[category] += 1
        difficulty_by_category[category][difficulty] += 1

    # Print difficulty distribution
    print("\n" + "="*80)
    print("📈 Difficulty Distribution (1=Very Easy, 5=Very Hard)")
    print("="*80)

    total = len(problems)
    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"  Level {difficulty}: {count:4d} ({percentage:5.1f}%) {bar}")

    # Print category distribution
    print("\n" + "="*80)
    print("📚 Category Distribution")
    print("="*80)

    # Sort by count (descending)
    sorted_categories = sorted(category_counts.items(), key=lambda x: -x[1])

    for category, count in sorted_categories:
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"  {category:45s}: {count:4d} ({percentage:5.1f}%) {bar}")

    # Print detailed breakdown
    print("\n" + "="*80)
    print("🔍 Difficulty by Category (Detailed Breakdown)")
    print("="*80)

    for category, count in sorted_categories:
        print(f"\n{category} (Total: {count})")
        difficulties = difficulty_by_category[category]
        for diff in sorted(difficulties.keys()):
            diff_count = difficulties[diff]
            diff_pct = (diff_count / count * 100) if count > 0 else 0
            print(f"  Level {diff}: {diff_count:3d} ({diff_pct:5.1f}%)")

    # Average difficulty by category
    print("\n" + "="*80)
    print("📊 Average Difficulty by Category")
    print("="*80)

    category_avg_difficulty = []
    for category, count in sorted_categories:
        difficulties = difficulty_by_category[category]
        total_diff = sum(diff * cnt for diff, cnt in difficulties.items() if isinstance(diff, (int, float)))
        avg_diff = total_diff / count if count > 0 else 0
        category_avg_difficulty.append((category, avg_diff, count))

    # Sort by average difficulty
    category_avg_difficulty.sort(key=lambda x: -x[1])

    for category, avg_diff, count in category_avg_difficulty:
        print(f"  {category:45s}: {avg_diff:.2f} (n={count})")

    # Summary statistics
    print("\n" + "="*80)
    print("📋 Summary")
    print("="*80)

    all_difficulties = [d for d in difficulty_counts.keys() if isinstance(d, (int, float))]
    if all_difficulties:
        total_weighted = sum(diff * difficulty_counts[diff] for diff in all_difficulties)
        avg_difficulty = total_weighted / total if total > 0 else 0
        min_difficulty = min(all_difficulties)
        max_difficulty = max(all_difficulties)

        print(f"  Average difficulty: {avg_difficulty:.2f}")
        print(f"  Difficulty range: {min_difficulty} - {max_difficulty}")
        print(f"  Number of categories: {len(category_counts)}")
        print(f"  Most common category: {sorted_categories[0][0]} ({sorted_categories[0][1]} problems)")
        print(f"  Least common category: {sorted_categories[-1][0]} ({sorted_categories[-1][1]} problems)")

    print("\n" + "="*80)


if __name__ == "__main__":
    dataset_path = "data/geoqa3_dataset.json"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    analyze_dataset(dataset_path)
