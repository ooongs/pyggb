#!/usr/bin/env python3
"""
Recalculate metrics and detailed_analysis from results in result.json.
Use this after reconstructing result.json to update aggregated metrics.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Import from benchmark runner
sys.path.insert(0, str(Path(__file__).parent))
from run_agent_benchmark import DetailedMetrics

def recalculate_metrics(result_path: Path):
    """Recalculate all metrics from results."""

    print(f"Recalculating metrics for: {result_path}")
    print()

    # Load result.json
    with open(result_path, 'r', encoding='utf-8') as f:
        result_json = json.load(f)

    results = result_json.get('results', [])
    print(f"📊 Found {len(results)} results to process")

    # Create backup
    backup_path = result_path.parent / f"result_backup_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Backup created: {backup_path}")
    print()

    # Create metrics calculator
    metrics = DetailedMetrics()

    print("🔄 Calculating metrics...")

    # Process each result
    for i, result in enumerate(results, 1):
        # Load memory data if available
        memory_data = None
        memory_path = result.get('memory_path')

        if memory_path and Path(memory_path).exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
            except Exception as e:
                print(f"  ⚠️  Could not load memory for {result.get('problem_id')}: {e}")

        # Handle skipped problems
        if result.get('skipped'):
            metrics.add_skipped_problem(
                problem_id=result['problem_id'],
                reason=result.get('skip_reason', 'Unknown'),
                error_types=result.get('validation_result', {}).get('dataset_error_types', [])
            )
        else:
            metrics.add_problem_result(result, memory_data)

            # Enhance problem_details with missing_objects and failed_conditions from memory
            if memory_data and 'steps' in memory_data and metrics.problem_details:
                # Find the problem_detail we just added
                problem_detail = metrics.problem_details[-1]

                # Extract final validation result from memory
                for step in reversed(memory_data['steps']):
                    validation = step.get('observation', {}).get('validation_result')
                    if validation:
                        # Add missing_objects to problem_detail
                        missing = validation.get('missing_objects', {})
                        problem_detail['missing_objects'] = missing
                        problem_detail['missing_objects_count'] = sum(
                            len(objs) if isinstance(objs, list) else 0
                            for objs in missing.values()
                        )

                        # Add failed_conditions to problem_detail
                        failed_conds = validation.get('failed_conditions', [])
                        problem_detail['failed_conditions'] = failed_conds
                        problem_detail['failed_conditions_count'] = len(failed_conds)
                        break

        if i % 50 == 0:
            print(f"  Processed {i}/{len(results)} results...")

    print(f"✓ Processed all {len(results)} results")
    print()

    # Get summary
    metrics_summary = metrics.get_summary()

    # Update result.json
    result_json['metrics'] = metrics_summary
    result_json['detailed_analysis'] = {
        "success_rate": {
            "value": metrics_summary["success_rate"],
            "successful": metrics_summary["successful_problems"],
            "failed": metrics_summary["failed_problems"],
            "skipped": metrics_summary["skipped_problems"],
            "total": metrics_summary["total_problems"],
            "note": "Success rate is calculated excluding skipped problems"
        },
        "step_analysis": {
            "average_steps_to_success": metrics_summary["average_success_steps"],
            "min_steps_to_success": metrics_summary["min_success_steps"],
            "max_steps_to_success": metrics_summary["max_success_steps"],
            "step_distribution": metrics_summary["success_step_distribution"]
        },
        "hallucination_analysis": {
            "total_hallucinations": metrics_summary["total_hallucinations"],
            "average_per_problem": metrics_summary["average_hallucinations_per_problem"],
            "average_recovery_steps": metrics_summary["average_hallucination_recovery_steps"],
            "error_type_distribution": metrics_summary["hallucination_error_types"]
        },
        "object_analysis": {
            "total_missing_objects": metrics_summary["total_missing_objects"],
            "missing_by_type": metrics_summary["missing_objects_by_type"]
        },
        "condition_analysis": {
            "total_failed_conditions": metrics_summary["total_failed_conditions"],
            "failed_by_type": metrics_summary["failed_conditions_by_type"]
        },
        "validation_error_analysis": {
            "total_errors": metrics_summary["validation_errors_count"],
            "errors_by_type": metrics_summary["validation_errors_by_type"],
            "unknown_condition_types": metrics_summary["unknown_condition_types"]
        }
    }

    result_json['problem_details'] = metrics.problem_details

    # Update metadata
    result_json['metadata']['metrics_recalculated'] = datetime.now().isoformat()
    result_json['metadata']['total_problems_evaluated'] = len(results)

    # Save updated result.json
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated result.json with recalculated metrics")
    print()

    # Print summary
    print("="*70)
    print("📊 METRICS SUMMARY")
    print("="*70)
    print()
    print(f"📈 Success Rate")
    print("-"*40)
    print(f"  Total Evaluated: {metrics_summary['total_problems']}")
    print(f"  Successful: {metrics_summary['successful_problems']} ({metrics_summary['success_rate']:.1%})")
    print(f"  Failed: {metrics_summary['failed_problems']}")
    if metrics_summary.get('skipped_problems', 0) > 0:
        print(f"  Skipped: {metrics_summary['skipped_problems']}")

    print()
    print(f"📉 Step Analysis")
    print("-"*40)
    print(f"  Average Steps to Success: {metrics_summary['average_success_steps']:.2f}")
    print(f"  Min Steps: {metrics_summary['min_success_steps']}")
    print(f"  Max Steps: {metrics_summary['max_success_steps']}")

    print()
    print(f"🔥 Hallucination Analysis")
    print("-"*40)
    print(f"  Total Hallucinations: {metrics_summary['total_hallucinations']}")
    print(f"  Average per Problem: {metrics_summary['average_hallucinations_per_problem']:.2f}")

    print()
    print("="*70)

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python recalculate_metrics.py <path_to_result.json>")
        print("Example: python recalculate_metrics.py agent_logs/gpt-5.1_no_vision_20260105_101316/result.json")
        sys.exit(1)

    result_path = Path(sys.argv[1])

    if not result_path.exists():
        print(f"❌ Error: File not found: {result_path}")
        sys.exit(1)

    success = recalculate_metrics(result_path)

    if success:
        print("✓ Done!")
    else:
        print("❌ Failed to recalculate metrics")
        sys.exit(1)

if __name__ == "__main__":
    main()
