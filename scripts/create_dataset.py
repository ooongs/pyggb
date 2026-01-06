#!/usr/bin/env python3
"""
Create Dataset from GeoQA3 JSON Files
Uses problem_parser.py to batch process problems with filtering and classification.

Supports:
- Incremental saving (append to file as each problem is parsed)
- Range-based parallel processing (e.g., 0-500, 500-1000)
- JSONL format for easy appending and merging

Usage:
    # Process all files
    python create_dataset.py
    
    # Process specific range (for parallel processing)
    python create_dataset.py --start 0 --end 500
    python create_dataset.py --start 500 --end 1000
    
    # Merge all partial files
    python create_dataset.py --merge
"""

import os
import sys
import glob
import json
import argparse
import fcntl
import time
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.problem_parser import ProblemParser, create_openai_api_function
from src.utils import get_data_dir, get_data_dir
from dotenv import load_dotenv

load_dotenv()


def get_json_files(input_dir: str, file_pattern: str = "*.json") -> list:
    """Get sorted list of JSON files in directory."""
    json_files = glob.glob(os.path.join(input_dir, file_pattern))
    # Sort by numeric ID in filename
    def extract_id(f):
        try:
            basename = os.path.basename(f).replace('.json', '')
            return int(basename)
        except:
            return float('inf')
    return sorted(json_files, key=extract_id)


def append_to_jsonl(output_file: str, data: dict):
    """
    Append a single JSON object to JSONL file with file locking.
    Thread-safe for parallel processing.
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        try:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock


def read_jsonl(file_path: str) -> list:
    """Read all JSON objects from JSONL file."""
    results = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def get_processed_ids(output_dir: str, pattern: str = "dataset_*.jsonl") -> set:
    """
    Get set of already processed problem IDs from all JSONL files in output directory.
    This enables resume across multiple parallel runs.
    """
    processed = set()
    jsonl_files = glob.glob(os.path.join(output_dir, pattern))
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if 'id' in data:
                                processed.add(str(data['id']))
                        except:
                            pass
        except:
            pass
    
    return processed


def process_range(parser: ProblemParser, json_files: list, output_file: str,
                  output_dir: str, start_idx: int, end_idx: int, 
                  skip_ambiguous: bool = True, clean_text: bool = True, 
                  resume: bool = True):
    """
    Process a range of files and save incrementally.
    Uses two-stage LLM processing:
    - Stage 1: Validate problem suitability + clean text
    - Stage 2: Extract objects/conditions + rate construction difficulty
    
    Args:
        parser: ProblemParser instance
        json_files: List of all JSON files
        output_file: Output JSONL file path
        output_dir: Output directory (for checking all processed IDs)
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        skip_ambiguous: Skip problems with ambiguous references
        clean_text: Ignored (cleaning is now part of Stage 1)
        resume: Skip already processed IDs
    """
    # Get files in range
    files_in_range = json_files[start_idx:end_idx]
    
    # Get already processed IDs from ALL JSONL files in output directory (for resume)
    processed_ids = get_processed_ids(output_dir) if resume else set()
    
    total = len(files_in_range)
    parsed_count = 0
    skipped_count = 0
    error_count = 0
    already_done = 0
    
    print(f"\n{'='*60}")
    print(f"Two-Stage LLM Processing")
    print(f"  Stage 1: Validate + Clean")
    print(f"  Stage 2: Parse + Rate Difficulty (construction-focused)")
    print(f"{'='*60}")
    print(f"Processing range: {start_idx} to {end_idx}")
    print(f"Files to process: {total}")
    print(f"Output file: {output_file}")
    print(f"Resume mode: {resume} (found {len(processed_ids)} already processed)")
    print(f"{'='*60}\n")
    
    for i, json_file in enumerate(files_in_range):
        # Extract problem ID from filename
        basename = os.path.basename(json_file).replace('.json', '')
        
        # Skip if already processed
        if basename in processed_ids:
            already_done += 1
            continue
        
        try:
            # Two-stage processing happens inside parse_from_json
            result = parser.parse_from_json(json_file, skip_ambiguous, clean_text)
            
            if result is None:
                skipped_count += 1
                # Save skip info for tracking
                skip_info = {
                    "id": basename,
                    "status": "skipped",
                    "reason": "validation_failed"
                }
                append_to_jsonl(output_file, skip_info)
                # Progress output for skipped
                progress = (i + 1) / total * 100
                print(f"[{progress:5.1f}%] ⊘ Skipped: {basename}")
            else:
                parsed_count += 1
                result["status"] = "parsed"
                append_to_jsonl(output_file, result)
                
                # Progress output
                progress = (i + 1) / total * 100
                print(f"[{progress:5.1f}%] ✓ Parsed: {basename}", end="")
                if 'category' in result:
                    cat = result['category'][:18] if len(result.get('category', '')) > 18 else result.get('category', 'N/A')
                    print(f" | {cat:18s} | Diff: {result.get('difficulty', '?')}/5")
                else:
                    print()
            
        except Exception as e:
            error_count += 1
            error_info = {
                "id": basename,
                "status": "error",
                "error": str(e)
            }
            append_to_jsonl(output_file, error_info)
            print(f"[{(i + 1) / total * 100:5.1f}%] ✗ Error: {basename} - {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Range {start_idx}-{end_idx} Complete!")
    print(f"{'='*60}")
    print(f"  Already done: {already_done}")
    print(f"  Parsed: {parsed_count}")
    print(f"  Skipped (validation failed): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total processed this run: {parsed_count + skipped_count + error_count}")
    
    return {
        "start": start_idx,
        "end": end_idx,
        "already_done": already_done,
        "parsed": parsed_count,
        "skipped": skipped_count,
        "errors": error_count
    }


def merge_jsonl_files(output_dir: str, final_output: str, pattern: str = "dataset_*.jsonl"):
    """
    Merge multiple JSONL files into a single JSON dataset.
    
    Args:
        output_dir: Directory containing JSONL files
        final_output: Final output JSON file path
        pattern: Glob pattern for JSONL files
    """
    print(f"\n{'='*60}")
    print("Merging JSONL files...")
    print(f"{'='*60}\n")
    
    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(output_dir, pattern))
    print(f"Found {len(jsonl_files)} JSONL files to merge")
    
    # Read all data
    all_data = []
    seen_ids = set()
    
    for jsonl_file in sorted(jsonl_files):
        print(f"  Reading: {jsonl_file}")
        data = read_jsonl(jsonl_file)
        for item in data:
            item_id = str(item.get('id', ''))
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                all_data.append(item)
    
    # Separate parsed, skipped, and errors
    parsed = [d for d in all_data if d.get('status') == 'parsed']
    skipped = [d for d in all_data if d.get('status') == 'skipped']
    errors = [d for d in all_data if d.get('status') == 'error']
    
    # Remove status field from parsed data
    for item in parsed:
        item.pop('status', None)
    
    # Create final dataset
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_problems": len(parsed),
            "skipped": len(skipped),
            "errors": len(errors),
            "skipped_ids": [d['id'] for d in skipped],
            "error_ids": [d['id'] for d in errors]
        },
        "problems": parsed
    }
    
    # Save final dataset
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"  Total parsed: {len(parsed)}")
    print(f"  Total skipped: {len(skipped)}")
    print(f"  Total errors: {len(errors)}")
    print(f"  Output: {final_output}")
    
    # Print category distribution if available
    if parsed and 'category' in parsed[0]:
        categories = {}
        difficulties = {}
        
        for problem in parsed:
            cat = problem.get('category', 'Unknown')
            diff = problem.get('difficulty', 0)
            
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print("\nCategory Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(parsed) * 100
            print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nDifficulty Distribution:")
        for diff, count in sorted(difficulties.items()):
            bar = "█" * (count // 10)
            print(f"  Level {diff}: {bar} ({count})")
    
    return dataset


def main():
    """Main function to create dataset from JSON files."""
    
    # Parse arguments
    parser_args = argparse.ArgumentParser(description="Create geometry problem dataset")
    default_input = str(get_data_dir() / "GeoQA3" / "json")
    default_output = str(get_data_dir())
    parser_args.add_argument("--input-dir", type=str, default=default_input,
                           help="Input directory containing JSON files")
    parser_args.add_argument("--output-dir", type=str, default=default_output,
                           help="Output directory for dataset files")
    parser_args.add_argument("--start", type=int, default=None,
                           help="Start index (inclusive)")
    parser_args.add_argument("--end", type=int, default=None,
                           help="End index (exclusive)")
    parser_args.add_argument("--merge", action="store_true",
                           help="Merge all JSONL files into final dataset")
    parser_args.add_argument("--no-resume", action="store_true",
                           help="Don't skip already processed IDs")
    parser_args.add_argument("--model", type=str, default="gpt-4o-mini",
                           help="OpenAI model to use")
    args = parser_args.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"✓ Created output directory: {args.output_dir}")
    
    # Handle merge mode
    if args.merge:
        final_output = os.path.join(args.output_dir, "geoqa3_dataset_new.json")
        merge_jsonl_files(args.output_dir, final_output)
        return
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        print("   Or add it to your ~/.zshrc or ~/.bashrc")
        print("\n   Without API key, classification and difficulty rating will not be available.")
        
        response = input("\nContinue without classification? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        
        llm_function = None
    else:
        print("✓ OpenAI API key found")
        print(f"  Using {args.model} for two-stage processing:")
        print(f"    - Stage 1: Problem validation + text cleaning")
        print(f"    - Stage 2: Object extraction + construction difficulty rating\n")
        llm_function = create_openai_api_function(model=args.model, api_key=api_key)
    
    # Initialize parser
    problem_parser = ProblemParser(llm_api_function=llm_function)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"✗ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Get all JSON files
    json_files = get_json_files(args.input_dir)
    total_files = len(json_files)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Total files found: {total_files}")
    
    # Determine range
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else total_files
    
    # Validate range
    start_idx = max(0, min(start_idx, total_files))
    end_idx = max(start_idx, min(end_idx, total_files))
    
    # Determine output file name based on range
    if args.start is not None or args.end is not None:
        output_file = os.path.join(args.output_dir, f"dataset_{start_idx}_{end_idx}.jsonl")
    else:
        output_file = os.path.join(args.output_dir, "dataset_all.jsonl")
    
    print(f"Output file: {output_file}")
    print(f"\nProcessing options:")
    print(f"  - Range: {start_idx} to {end_idx} ({end_idx - start_idx} files)")
    print(f"  - Skip ambiguous references (∠1, ∠2, etc.): ✓")
    print(f"  - Clean text (remove '如图所示', questions): ✓")
    print(f"  - Classify problems: {'✓' if llm_function else '✗'}")
    print(f"  - Rate difficulty: {'✓' if llm_function else '✗'}")
    print(f"  - Resume mode: {'✗' if args.no_resume else '✓'}")
    
    # Process range
    stats = process_range(
        parser=problem_parser,
        json_files=json_files,
        output_file=output_file,
        output_dir=args.output_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        skip_ambiguous=True,
        clean_text=True,
        resume=not args.no_resume
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Output saved to: {output_file}")
    print(f"\n  To merge all files into final dataset:")
    print(f"  python create_dataset.py --merge")


if __name__ == "__main__":
    main()



