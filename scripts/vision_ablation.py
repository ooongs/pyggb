#!/usr/bin/env python3
"""Run a vision ablation benchmark on GeoQA3.

This script runs the agent benchmark twice on the same dataset:
  1) vision enabled (default behavior)
  2) vision disabled (--no-vision)

Outputs are written to benchmark_results/vision_ablation_<timestamp>/.
It also generates comparison reports (text, csv, json) using the existing
analyze_benchmark_results utility.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "geoqa3_dataset.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run vision/no-vision ablation on GeoQA3"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to benchmark (used when --models is not provided)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to benchmark (e.g. qwen/qwen2.5-vl-72b-instruct opengvlab/internvl3-78b)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="Solve a single problem by ID (disables --batch mode options)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum reasoning iterations per problem",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems (default: all)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index within the dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store ablation outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose through to the benchmark runner",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug through to the benchmark runner",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving intermediate images during benchmarking",
    )
    parser.add_argument(
        "--additional-prompt",
        type=Path,
        default=None,
        help="Path to text file appended to the system prompt",
    )
    parser.add_argument(
        "--resume-memory",
        type=Path,
        default=None,
        help="Path to memory JSON file to resume from (single-problem mode only)",
    )
    parser.add_argument(
        "--resume-from-iteration",
        type=int,
        default=None,
        help="Resume from the next iteration after this step (requires --resume-memory)",
    )
    parser.add_argument(
        "--reexecute-last",
        action="store_true",
        help="Re-execute DSL from the last loaded iteration (requires --resume-memory)",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Run only the vision-enabled pass (skip no-vision)",
    )
    parser.add_argument(
        "--no-vision-only",
        action="store_true",
        help="Run only the no-vision pass (skip vision-enabled)",
    )
    return parser.parse_args()


def build_env_for_runs() -> Dict[str, str]:
    """Build environment with OpenRouter fallbacks for OpenAI-compatible client."""
    env = os.environ.copy()
    openrouter_key = env.get("OPENROUTER_API_KEY")
    openrouter_base = env.get("OPENROUTER_API_BASE") or (OPENROUTER_DEFAULT_BASE if openrouter_key else None)

    if openrouter_key:
        env.setdefault("OPENAI_API_KEY", openrouter_key)
    if openrouter_base:
        env.setdefault("OPENAI_API_BASE", openrouter_base)
        env.setdefault("OPENROUTER_API_BASE", openrouter_base)

    return env


def run_command(cmd: List[str], log_path: Path, env: Dict[str, str]) -> None:
    """Run a command, streaming stdout to both console and log."""
    printable_cmd = " ".join(cmd)
    print(f"\n$ {printable_cmd}")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {printable_cmd}\n")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def build_benchmark_cmd(use_vision: bool, output_path: Path, args: argparse.Namespace) -> List[str]:
    """Construct the run_agent_benchmark.py command."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_agent_benchmark.py"),
        "--model",
        args.model,
        "--dataset",
        str(args.dataset),
        "--max-iter",
        str(args.max_iter),
        "--output",
        str(output_path),
    ]
    if args.problem_id:
        cmd += ["--problem-id", args.problem_id]
    else:
        cmd += ["--batch", "--start-idx", str(args.start_idx)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.verbose:
        cmd.append("--verbose")
    if args.debug:
        cmd.append("--debug")
    if args.no_save_images:
        cmd.append("--no-save-images")
    if args.additional_prompt:
        cmd += ["--additional-prompt", str(args.additional_prompt)]
    if args.resume_memory:
        cmd += ["--resume-memory", str(args.resume_memory)]
    if args.resume_from_iteration is not None:
        cmd += ["--resume-from-iteration", str(args.resume_from_iteration)]
    if args.reexecute_last:
        cmd.append("--reexecute-last")
    if not use_vision:
        cmd.append("--no-vision")
    return cmd


def collect_metrics(result_path: Path) -> Dict[str, float]:
    """Extract the key metrics we want to summarize."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict) and isinstance(metrics.get("summary"), dict):
        metrics = metrics["summary"]
    return {
        "success_rate": metrics.get("success_rate", 0.0),
        "average_success_steps": metrics.get("average_success_steps", 0.0),
        "average_hallucinations_per_problem": metrics.get("average_hallucinations_per_problem", 0.0),
        "total_missing_objects": metrics.get("total_missing_objects", 0),
        "total_failed_conditions": metrics.get("total_failed_conditions", 0),
        "total_problems": metrics.get("total_problems", 0),
        "successful_problems": metrics.get("successful_problems", 0),
        "failed_problems": metrics.get("failed_problems", 0),
        "skipped_problems": metrics.get("skipped_problems", 0),
    }


def write_summary(
    with_vision: Dict[str, float],
    no_vision: Dict[str, float],
    summary_path: Path,
) -> None:
    """Write a simple text summary comparing vision vs no-vision."""
    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = []
    lines.append("Vision Ablation Summary")
    lines.append("-" * 72)
    lines.append(f"Results file: {summary_path.parent}")
    lines.append(f"Problems evaluated (vision):   {with_vision['total_problems']} "
                 f"(skipped: {with_vision['skipped_problems']})")
    lines.append(f"Problems evaluated (no vision): {no_vision['total_problems']} "
                 f"(skipped: {no_vision['skipped_problems']})")
    lines.append("")
    header = f"{'Metric':<32}{'With vision':>15}{'No vision':>15}{'Diff':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"{'Success rate':<32}"
        f"{pct(with_vision['success_rate']):>15}"
        f"{pct(no_vision['success_rate']):>15}"
        f"{(with_vision['success_rate'] - no_vision['success_rate']) * 100:>11.1f} pp"
    )
    lines.append(
        f"{'Avg steps to success':<32}"
        f"{with_vision['average_success_steps']:>15.2f}"
        f"{no_vision['average_success_steps']:>15.2f}"
        f"{with_vision['average_success_steps'] - no_vision['average_success_steps']:>12.2f}"
    )
    lines.append(
        f"{'Avg hallucinations/problem':<32}"
        f"{with_vision['average_hallucinations_per_problem']:>15.2f}"
        f"{no_vision['average_hallucinations_per_problem']:>15.2f}"
        f"{with_vision['average_hallucinations_per_problem'] - no_vision['average_hallucinations_per_problem']:>12.2f}"
    )
    lines.append(
        f"{'Total missing objects':<32}"
        f"{with_vision['total_missing_objects']:>15d}"
        f"{no_vision['total_missing_objects']:>15d}"
        f"{with_vision['total_missing_objects'] - no_vision['total_missing_objects']:>12d}"
    )
    lines.append(
        f"{'Total failed conditions':<32}"
        f"{with_vision['total_failed_conditions']:>15d}"
        f"{no_vision['total_failed_conditions']:>15d}"
        f"{with_vision['total_failed_conditions'] - no_vision['total_failed_conditions']:>12d}"
    )

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text + "\n")


def write_single_summary(metrics: Dict[str, float], summary_path: Path, label: str) -> None:
    """Write summary when only one mode was run."""
    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = []
    lines.append(f"{label} Summary")
    lines.append("-" * 72)
    lines.append(f"Results file: {summary_path.parent}")
    lines.append(f"Problems evaluated: {metrics['total_problems']} (skipped: {metrics['skipped_problems']})")
    lines.append("")
    lines.append(f"Success rate: {pct(metrics['success_rate'])}")
    lines.append(f"Avg steps to success: {metrics['average_success_steps']:.2f}")
    lines.append(f"Avg hallucinations/problem: {metrics['average_hallucinations_per_problem']:.2f}")
    lines.append(f"Total missing objects: {metrics['total_missing_objects']}")
    lines.append(f"Total failed conditions: {metrics['total_failed_conditions']}")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text + "\n")


def main() -> int:
    args = parse_args()
    env = build_env_for_runs()

    if args.vision_only and args.no_vision_only:
        raise SystemExit("Cannot use both --vision-only and --no-vision-only")
    if args.problem_id and (args.limit is not None or args.start_idx != 0):
        raise SystemExit("--problem-id cannot be combined with --limit or --start-idx")
    if (args.resume_memory or args.resume_from_iteration is not None or args.reexecute_last) and not args.problem_id:
        raise SystemExit("Resume options require --problem-id (single-problem mode)")
    if args.resume_from_iteration is not None and not args.resume_memory:
        raise SystemExit("--resume-from-iteration requires --resume-memory")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"vision_ablation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "ablation.log"

    models = args.models or [args.model]

    print(f"Running vision ablation into {run_dir}")
    print(f"Models: {', '.join(models)}")

    analyzer_script = REPO_ROOT / "scripts" / "analyze_benchmark_results.py"

    for model in models:
        print(f"\n=== Benchmarking {model} ===")
        args.model = model  # reuse arg container
        model_safe = model.replace("/", "__").replace(":", "__")

        run_with_vision = not args.no_vision_only
        run_no_vision = not args.vision_only

        with_vision_output = None
        no_vision_output = None

        # Run with vision
        if run_with_vision:
            with_vision_output = run_dir / f"results_{model_safe}_with_vision.json"
            run_command(build_benchmark_cmd(True, with_vision_output, args), log_path, env)

        # Run without vision
        if run_no_vision:
            no_vision_output = run_dir / f"results_{model_safe}_no_vision.json"
            run_command(build_benchmark_cmd(False, no_vision_output, args), log_path, env)

        # Generate reports
        if run_with_vision and run_no_vision and with_vision_output and no_vision_output:
            run_command(
                [
                    sys.executable,
                    str(analyzer_script),
                    str(with_vision_output),
                    str(no_vision_output),
                    "--format",
                    "text",
                    "--output",
                    str(run_dir / f"vision_ablation_report_{model_safe}.txt"),
                    "--per-problem",
                ],
                log_path,
                env,
            )
            run_command(
                [
                    sys.executable,
                    str(analyzer_script),
                    str(with_vision_output),
                    str(no_vision_output),
                    "--format",
                    "csv",
                    "--output",
                    str(run_dir / f"vision_ablation_report_{model_safe}.csv"),
                ],
                log_path,
                env,
            )
            run_command(
                [
                    sys.executable,
                    str(analyzer_script),
                    str(with_vision_output),
                    str(no_vision_output),
                    "--format",
                    "json",
                    "--output",
                    str(run_dir / f"vision_ablation_report_{model_safe}.json"),
                ],
                log_path,
                env,
            )

            # Comparison summary
            metrics_with = collect_metrics(with_vision_output)
            metrics_without = collect_metrics(no_vision_output)
            write_summary(
                metrics_with,
                metrics_without,
                run_dir / f"vision_ablation_summary_{model_safe}.txt",
            )
        elif run_with_vision and with_vision_output:
            metrics_with = collect_metrics(with_vision_output)
            write_single_summary(
                metrics_with,
                run_dir / f"vision_ablation_summary_{model_safe}.txt",
                "Vision-only",
            )
        elif run_no_vision and no_vision_output:
            metrics_without = collect_metrics(no_vision_output)
            write_single_summary(
                metrics_without,
                run_dir / f"vision_ablation_summary_{model_safe}.txt",
                "No-vision-only",
            )

    print(f"Artifacts written to: {run_dir}")
    print(f"Log file: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
