#!/usr/bin/env python3
"""
Reconstruct result.json from run_info.json and memory/log files.
This recovers result.json when merge failed during resume.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

def load_memory_file(memory_path: str) -> Optional[Dict]:
    """Load memory JSON file if it exists."""
    if not Path(memory_path).exists():
        return None

    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Could not load memory: {e}")
        return None

def reconstruct_result_from_memory(problem_info: Dict, memory_data: Dict,
                                   run_dir: Path, run_id: str) -> Dict[str, Any]:
    """Reconstruct a result entry from memory data."""
    problem_id = problem_info['problem_id']

    # Extract info from memory
    steps = memory_data.get('steps', [])
    iterations = len(steps)

    # Find final validation result
    validation_result = None
    final_dsl = None
    success = problem_info.get('success', False)

    for step in reversed(steps):
        obs = step.get('observation', {})
        if obs.get('validation_result'):
            validation_result = obs['validation_result']
            break

    # Get final DSL code
    if steps:
        last_action = steps[-1].get('action', {})
        final_dsl = last_action.get('dsl_code', '')

    # Construct paths
    session_id = f"{problem_id}_{run_id}"
    log_file = str(run_dir / "sessions" / f"{problem_id}.log")
    memory_path = f"agent_images/{problem_id}_memory.json"

    # Find images
    images = []
    images_dir = run_dir / "images"
    if images_dir.exists():
        for img in sorted(images_dir.glob(f"{problem_id}_iter*.png")):
            images.append(str(img))

    # Create summary
    start_time = problem_info.get('start_time', '')
    end_time = problem_info.get('end_time', '')

    duration_seconds = 0
    if start_time and end_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration_seconds = (end_dt - start_dt).total_seconds()
        except:
            pass

    # Count successful executions (DSL executions that didn't error)
    successful_executions = sum(
        1 for step in steps
        if step.get('observation', {}).get('success', False)
    )

    summary = {
        "duration_seconds": duration_seconds,
        "successful_executions": successful_executions,
        "total_iterations": iterations
    }

    # Build result entry
    result = {
        "problem_id": problem_id,
        "session_id": session_id,
        "success": success,
        "iterations": iterations,
        "log_file": log_file,
        "memory_path": memory_path,
        "final_dsl": final_dsl,
        "validation_result": validation_result,
        "images": images,
        "summary": summary
    }

    return result

def reconstruct_result_json(run_dir: Path, current_result_path: Path):
    """Reconstruct full result.json from run_info and memory files."""

    print(f"Reconstructing result.json for: {run_dir}")
    print()

    # Load run_info.json
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        print(f"❌ Error: run_info.json not found at {run_info_path}")
        return False

    with open(run_info_path, 'r', encoding='utf-8') as f:
        run_info = json.load(f)

    problems = run_info.get('problems', [])
    run_id = run_info.get('run_id', '')

    print(f"📋 Found {len(problems)} problems in run_info.json")

    # Load current result.json to preserve entries
    current_results = {}
    if current_result_path.exists():
        with open(current_result_path, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
            for result in current_data.get('results', []):
                pid = result.get('problem_id')
                if pid:
                    current_results[pid] = result
        print(f"📋 Found {len(current_results)} existing results in result.json")

    # Backup current result.json
    if current_result_path.exists():
        backup_path = current_result_path.parent / f"result_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(current_result_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Backup created: {backup_path}")

    print()
    print("🔄 Reconstructing results from memory files...")

    # Reconstruct results
    reconstructed = {}
    missing_memory = []

    for i, problem_info in enumerate(problems, 1):
        problem_id = problem_info['problem_id']

        # Skip if already in current results (prefer existing detailed results)
        if problem_id in current_results:
            reconstructed[problem_id] = current_results[problem_id]
            continue

        # Try to load memory file
        memory_path = f"agent_images/{problem_id}_memory.json"
        memory_data = load_memory_file(memory_path)

        if memory_data:
            result = reconstruct_result_from_memory(problem_info, memory_data, run_dir, run_id)
            reconstructed[problem_id] = result
            if i % 50 == 0:
                print(f"  Processed {i}/{len(problems)} problems...")
        else:
            missing_memory.append(problem_id)
            # Create minimal result entry from run_info only
            result = {
                "problem_id": problem_id,
                "session_id": f"{problem_id}_{run_id}",
                "success": problem_info.get('success', False),
                "iterations": problem_info.get('iterations', 0),
                "log_file": str(run_dir / "sessions" / f"{problem_id}.log"),
                "memory_path": memory_path,
                "final_dsl": None,
                "validation_result": None,
                "images": [],
                "summary": {
                    "duration_seconds": 0,
                    "successful_executions": 0,
                    "total_iterations": problem_info.get('iterations', 0)
                }
            }
            reconstructed[problem_id] = result

    print(f"✓ Reconstructed {len(reconstructed)} results")
    print(f"  - From memory files: {len(reconstructed) - len(missing_memory)}")
    print(f"  - From existing result.json: {len(current_results)}")
    print(f"  - Minimal entries (no memory): {len(missing_memory)}")

    if missing_memory:
        print(f"\n⚠️  Missing memory files for {len(missing_memory)} problems")
        if len(missing_memory) <= 10:
            print(f"    {missing_memory}")
        else:
            print(f"    {missing_memory[:10]} ... and {len(missing_memory)-10} more")

    # Load original result.json structure
    if current_result_path.exists():
        with open(current_result_path, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
    else:
        # Create minimal structure
        result_json = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "run_dir": str(run_dir)
            },
            "metrics": {},
            "detailed_analysis": {},
            "results": [],
            "problem_details": []
        }

    # Update results
    result_json['results'] = list(reconstructed.values())

    # Update metadata
    result_json['metadata']['last_updated'] = datetime.now().isoformat()
    result_json['metadata']['reconstructed'] = True
    result_json['metadata']['total_problems_evaluated'] = len(reconstructed)

    # Save reconstructed result.json
    with open(current_result_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print()
    print(f"✅ Reconstructed result.json saved to: {current_result_path}")
    print(f"   Total results: {len(reconstructed)}")
    print()
    print("⚠️  Note: Metrics and detailed_analysis need to be recalculated.")
    print("   Run the benchmark again with --resume to update metrics.")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_result_json.py <path_to_log_directory>")
        print("Example: python reconstruct_result_json.py agent_logs/gpt-5.1_no_vision_20260105_101316")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"❌ Error: Directory not found: {run_dir}")
        sys.exit(1)

    result_path = run_dir / "result.json"

    success = reconstruct_result_json(run_dir, result_path)

    if success:
        print("✓ Done!")
    else:
        print("❌ Failed to reconstruct result.json")
        sys.exit(1)

if __name__ == "__main__":
    main()
