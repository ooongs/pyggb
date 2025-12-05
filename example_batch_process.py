#!/usr/bin/env python3
"""
Example: Batch process a small sample of problems to demonstrate all features.
This script processes the first 20 JSON files as a demo.
"""

import os
import json
import shutil
from problem_parser import ProblemParser, create_openai_api_function


def create_sample_dataset():
    """Create a sample dataset from first 20 files."""
    
    # Configuration
    source_dir = "data-5/GeoQA3/json"
    sample_dir = "sample_data"
    output_file = "sample_dataset.json"
    num_files = 20
    
    print("="*70)
    print("Sample Batch Processing Demo")
    print("="*70)
    print(f"\nProcessing first {num_files} files from {source_dir}")
    print(f"Output will be saved to: {output_file}\n")
    
    # Create sample directory with first N files
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"✓ Created sample directory: {sample_dir}")
    
    # Copy first N files
    json_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.json')])[:num_files]
    
    for fname in json_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(sample_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    print(f"✓ Prepared {len(json_files)} sample files\n")
    
    # Setup parser
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ Using OpenAI API for classification and difficulty rating")
        llm_func = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
        parser = ProblemParser(llm_api_function=llm_func)
    else:
        print("⚠️  No OpenAI API key - skipping classification/difficulty rating")
        print("   Set with: export OPENAI_API_KEY='your-key'\n")
        parser = ProblemParser()
    
    print("\n" + "="*70)
    print("Starting batch processing...")
    print("="*70 + "\n")
    
    # Batch process
    stats = parser.batch_parse_directory(
        input_dir=sample_dir,
        output_file=output_file,
        skip_ambiguous=True,
        clean_text=True,
        file_pattern="*.json"
    )
    
    # Show some results
    print("\n" + "="*70)
    print("Sample Results Preview")
    print("="*70 + "\n")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Total problems processed: {len(dataset['problems'])}")
    
    if dataset['problems']:
        print("\nFirst problem example:")
        first = dataset['problems'][0]
        print(f"  ID: {first['id']}")
        print(f"  Original: {first.get('original_text', 'N/A')[:60]}...")
        print(f"  Cleaned:  {first.get('cleaned_text', 'N/A')[:60]}...")
        if 'category' in first:
            print(f"  Category: {first['category']}")
            print(f"  Difficulty: {first['difficulty']}/5")
        print(f"  Points: {first['required_objects']['points']}")
        print(f"  Conditions: {len(first['verification_conditions'])}")
    
    # Show statistics
    if dataset['problems'] and 'category' in dataset['problems'][0]:
        print("\n" + "-"*70)
        print("Category Distribution:")
        print("-"*70)
        
        categories = {}
        difficulties = {}
        
        for problem in dataset['problems']:
            cat = problem.get('category', 'Unknown')
            diff = problem.get('difficulty', 0)
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(dataset['problems']) * 100
            print(f"  {cat:40s}: {count:2d} ({percentage:5.1f}%)")
        
        print("\n" + "-"*70)
        print("Difficulty Distribution:")
        print("-"*70)
        
        for diff in sorted(difficulties.keys()):
            count = difficulties[diff]
            bar = "█" * count
            print(f"  Level {diff}: {bar} ({count})")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ Sample data in: {sample_dir}/")
    print(f"\nYou can now:")
    print(f"  1. Review the output: cat {output_file}")
    print(f"  2. Delete sample data: rm -rf {sample_dir}/")
    print(f"  3. Process full dataset: python create_dataset.py")


if __name__ == "__main__":
    create_sample_dataset()


