#!/usr/bin/env python3
"""
Resume or retry benchmark runs.

Features:
- Resume from specific problem ID
- Retry failed/incomplete problems from existing run
- Update existing benchmark results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from run_agent_benchmark import AgentBenchmarkRunner
from src.benchmark.benchmark_dataset import BenchmarkDataset


def find_incomplete_problems(log_dir: Path) -> List[str]:
    """Find problems that started but didn't complete (missing end_time)."""
    run_info_path = log_dir / "run_info.json"

    if not run_info_path.exists():
        print(f"❌ No run_info.json found in {log_dir}")
        return []

    with open(run_info_path) as f:
        run_info = json.load(f)

    incomplete = []
    for problem_info in run_info.get("problems", []):
        # Has start_time but no end_time
        if "start_time" in problem_info and "end_time" not in problem_info:
            incomplete.append(problem_info["problem_id"])

    return incomplete


def find_failed_problems(log_dir: Path) -> List[str]:
    """Find problems that failed (has end_time but success=False)."""
    run_info_path = log_dir / "run_info.json"

    if not run_info_path.exists():
        return []

    with open(run_info_path) as f:
        run_info = json.load(f)

    failed = []
    for problem_info in run_info.get("problems", []):
        # Has end_time and success is False
        if "end_time" in problem_info and not problem_info.get("success", True):
            failed.append(problem_info["problem_id"])

    return failed


def get_completed_problems(log_dir: Path) -> Set[str]:
    """Get set of successfully completed problem IDs."""
    run_info_path = log_dir / "run_info.json"

    if not run_info_path.exists():
        return set()

    with open(run_info_path) as f:
        run_info = json.load(f)

    completed = set()
    for problem_info in run_info.get("problems", []):
        # Has end_time and success is True
        if "end_time" in problem_info and problem_info.get("success", True):
            completed.add(problem_info["problem_id"])

    return completed


def get_log_dir_info(log_dir: Path) -> Dict:
    """Extract model name and configuration from log directory."""
    # Try to parse from directory name
    dir_name = log_dir.name

    # Format: {model}_[vision_]{timestamp}
    # e.g., gpt-4o_vision_20251219_160112
    parts = dir_name.split("_")

    info = {
        "model": None,
        "use_vision": False,
        "timestamp": None,
    }

    # Find timestamp (last part, format: YYYYMMDD_HHMMSS)
    if len(parts) >= 2:
        potential_timestamp = "_".join(parts[-2:])
        if len(potential_timestamp.replace("_", "")) == 14:
            info["timestamp"] = potential_timestamp
            parts = parts[:-2]

    # Check for vision flag
    if parts and parts[-1] == "vision":
        info["use_vision"] = True
        parts = parts[:-1]

    # Remaining is model name
    if parts:
        info["model"] = "_".join(parts)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Resume or retry benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume from problem 239 in existing log directory
  python resume_benchmark.py --log-dir agent_logs/gpt-4o_vision_20251219_160112 --from-id 239

  # Retry all incomplete problems (crashed/interrupted)
  python resume_benchmark.py --log-dir agent_logs/gpt-4o_vision_20251219_160112 --retry-incomplete

  # Retry all failed problems
  python resume_benchmark.py --log-dir agent_logs/gpt-4o_vision_20251219_160112 --retry-failed

  # Retry specific problem IDs
  python resume_benchmark.py --log-dir agent_logs/gpt-4o_vision_20251219_160112 --problem-ids 10,25,30

  # Resume from ID 100 and skip already completed problems
  python resume_benchmark.py --log-dir agent_logs/gpt-4o_vision_20251219_160112 --from-id 100 --skip-completed
        """
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Existing agent_logs directory to resume/update"
    )

    parser.add_argument(
        "--from-id",
        type=str,
        help="Resume from this problem ID onwards"
    )

    parser.add_argument(
        "--problem-ids",
        type=str,
        help="Comma-separated list of specific problem IDs to run"
    )

    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Retry problems that started but didn't complete"
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry problems that failed"
    )

    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip already completed problems (useful with --from-id)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/geoqa3_dataset.json",
        help="Dataset file (default: data/geoqa3_dataset.json)"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum iterations per problem (default: 5)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Override model name (default: auto-detect from log dir)"
    )

    parser.add_argument(
        "--use-vision",
        type=bool,
        help="Enable vision mode (default: auto-detect from log dir)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Auto-detect model and settings from log directory
    log_info = get_log_dir_info(log_dir)
    model = args.model or log_info["model"]
    use_vision = args.use_vision 
    use_vision = False

    if not model:
        print(f"❌ Error: Could not detect model from directory name. Please specify --model")
        sys.exit(1)

    print(f"📁 Log directory: {log_dir}")
    print(f"🤖 Model: {model}")
    print(f"👁️  Vision: {'enabled' if use_vision else 'disabled'}")
    print()

    # Determine which problems to run
    problem_ids_to_run = []

    if args.retry_incomplete:
        incomplete = find_incomplete_problems(log_dir)
        print(f"🔍 Found {len(incomplete)} incomplete problems")
        problem_ids_to_run.extend(incomplete)

    if args.retry_failed:
        failed = find_failed_problems(log_dir)
        print(f"🔍 Found {len(failed)} failed problems")
        problem_ids_to_run.extend(failed)

    if args.problem_ids:
        specified = args.problem_ids.split(",")
        print(f"🔍 Specified {len(specified)} problem IDs")
        problem_ids_to_run.extend(specified)

    # Load dataset
    dataset = BenchmarkDataset(args.dataset)
    all_problems = {p.id: p for p in dataset.problems}

    if args.from_id:
        # Get all problems from this ID onwards
        from_idx = None
        for idx, problem in enumerate(dataset.problems):
            if problem.id == args.from_id:
                from_idx = idx
                break

        if from_idx is None:
            print(f"❌ Error: Problem ID '{args.from_id}' not found in dataset")
            sys.exit(1)

        remaining_problems = [p.id for p in dataset.problems[from_idx:]]
        print(f"🔍 Resuming from problem {args.from_id}: {len(remaining_problems)} problems")

        if args.skip_completed:
            completed = get_completed_problems(log_dir)
            before_count = len(remaining_problems)
            remaining_problems = [pid for pid in remaining_problems if pid not in completed]
            skipped = before_count - len(remaining_problems)
            print(f"⏭️  Skipped {skipped} already completed problems")

        problem_ids_to_run.extend(remaining_problems)

    # Remove duplicates while preserving order
    seen = set()
    problem_ids_to_run = [pid for pid in problem_ids_to_run if not (pid in seen or seen.add(pid))]

    if not problem_ids_to_run:
        print("ℹ️  No problems to run!")
        sys.exit(0)

    print(f"\n🚀 Running {len(problem_ids_to_run)} problems")
    print(f"   First: {problem_ids_to_run[0]}")
    print(f"   Last: {problem_ids_to_run[-1]}")
    print()

    # Filter dataset to only include specified problems
    problems_to_run = [all_problems[pid] for pid in problem_ids_to_run if pid in all_problems]

    if len(problems_to_run) != len(problem_ids_to_run):
        missing = set(problem_ids_to_run) - set(p.id for p in problems_to_run)
        print(f"⚠️  Warning: {len(missing)} problem IDs not found in dataset: {missing}")

    # Run benchmark (use absolute path to ensure merge works)
    result_file = (log_dir / "result.json").resolve()

    try:
        # Create runner with resume_dir
        runner = AgentBenchmarkRunner(
            model=model,
            max_iterations=args.max_iter,
            save_images=True,
            verbose=args.verbose,
            use_vision=use_vision,
            resume_dir=str(log_dir)
        )

        # Run batch with filtered problems
        report = runner.run_batch(
            dataset_path=args.dataset,
            output_file=str(result_file),
            problems_filter=problems_to_run
        )

        print("\n✅ Benchmark resumed successfully!")
        print(f"📊 Results updated in: {result_file}")

        # Print summary
        metrics = report.get("metrics", {})
        print(f"\n📈 Summary:")
        print(f"   Total: {metrics.get('total_problems', 0)}")
        print(f"   Success: {metrics.get('successful_problems', 0)}")
        print(f"   Failed: {metrics.get('failed_problems', 0)}")
        print(f"   Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"💾 Progress saved in: {log_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
