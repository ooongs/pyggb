#!/usr/bin/env python3
"""
Run Agent Benchmark
Main script for evaluating ReAct agent on geometry problems.
Includes detailed metrics for model comparison and analysis.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from benchmark_dataset import BenchmarkDataset
from react_agent import ReActAgent
from dsl_validator import ValidationErrorLogger, set_validation_error_logger


class DetailedMetrics:
    """Collects detailed metrics for analysis."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_problems = 0
        self.successful_problems = 0
        self.failed_problems = 0
        self.skipped_problems = 0  # Problems skipped due to dataset errors
        
        # Step-level metrics
        self.success_steps = []  # Steps taken to succeed for each problem
        self.total_steps_per_problem = []
        
        # Hallucination metrics (DSL execution errors)
        self.hallucination_counts = []  # DSL errors per problem
        self.hallucination_recovery_steps = []  # Steps to recover from hallucination
        self.hallucination_error_types = {}  # Error type distribution
        
        # Object missing metrics
        self.missing_objects_counts = []  # Missing objects per problem
        self.missing_objects_by_type = {
            "points": 0,
            "segments": 0,
            "lines": 0,
            "circles": 0,
            "polygons": 0
        }
        
        # Condition failure metrics
        self.failed_conditions_counts = []  # Failed conditions per problem
        self.failed_conditions_by_type = {}  # Condition type distribution
        
        # Validation error metrics (dataset/condition errors)
        self.validation_errors = []  # List of validation errors
        self.validation_errors_by_type = {}  # Error type distribution
        self.unknown_condition_types = set()  # Unknown condition types encountered
        
        # Skipped problems tracking
        self.skipped_problem_ids = []  # IDs of skipped problems
        self.skipped_reasons = {}  # Reason for each skipped problem
        
        # Per-problem detailed records
        self.problem_details = []
    
    def add_problem_result(self, result: Dict[str, Any], memory_data: Optional[Dict] = None):
        """Add a single problem result to metrics."""
        self.total_problems += 1
        
        problem_detail = {
            "problem_id": result.get("problem_id"),
            "success": result.get("success", False),
            "iterations": result.get("iterations", 0),
            "hallucinations": [],
            "hallucination_count": 0,
            "missing_objects": {},
            "missing_objects_count": 0,
            "failed_conditions": [],
            "failed_conditions_count": 0,
            "success_step": None,
            "object_score": 0.0,
            "condition_score": 0.0,
            "total_score": 0.0
        }
        
        if result.get("success"):
            self.successful_problems += 1
            problem_detail["success_step"] = result.get("iterations", 0)
            self.success_steps.append(result.get("iterations", 0))
        else:
            self.failed_problems += 1
        
        self.total_steps_per_problem.append(result.get("iterations", 0))
        
        # Extract detailed metrics from memory if available
        if memory_data and "steps" in memory_data:
            hallucination_count = 0
            hallucination_errors = []
            last_error_step = None
            recovery_steps = []
            
            for step in memory_data["steps"]:
                obs = step.get("observation", {})
                
                # Count DSL execution errors (hallucinations)
                if not obs.get("success", True) and obs.get("error"):
                    hallucination_count += 1
                    error_msg = obs.get("error", "")
                    error_type = self._categorize_error(error_msg)
                    
                    hallucination_errors.append({
                        "step": step.get("iteration"),
                        "error_type": error_type,
                        "error_message": error_msg[:200]
                    })
                    
                    # Track error types
                    self.hallucination_error_types[error_type] = \
                        self.hallucination_error_types.get(error_type, 0) + 1
                    
                    last_error_step = step.get("iteration")
                
                # Track recovery from hallucination
                elif obs.get("success", False) and last_error_step is not None:
                    recovery = step.get("iteration") - last_error_step
                    recovery_steps.append(recovery)
                    last_error_step = None
                
                # Extract validation results
                validation = obs.get("validation_result")
                if validation:
                    problem_detail["object_score"] = validation.get("object_score", 0.0)
                    problem_detail["condition_score"] = validation.get("condition_score", 0.0)
                    problem_detail["total_score"] = validation.get("total_score", 0.0)
                    
                    # Missing objects
                    missing = validation.get("missing_objects", {})
                    for obj_type, objs in missing.items():
                        if objs:
                            self.missing_objects_by_type[obj_type] = \
                                self.missing_objects_by_type.get(obj_type, 0) + len(objs)
                    
                    # Failed conditions
                    failed_conds = validation.get("failed_conditions", [])
                    for cond in failed_conds:
                        cond_type = cond.get("type", "unknown")
                        self.failed_conditions_by_type[cond_type] = \
                            self.failed_conditions_by_type.get(cond_type, 0) + 1
                        
                        # Check for validation errors (unknown conditions, dataset errors)
                        error_type = cond.get("error_type")
                        if error_type in ["unknown_condition", "execution_error", "dataset_error"]:
                            self.validation_errors.append({
                                "problem_id": result.get("problem_id"),
                                "condition_type": cond_type,
                                "condition_data": cond,
                                "error_type": error_type,
                                "validation_message": cond.get("validation_message", "")
                            })
                            self.validation_errors_by_type[error_type] = \
                                self.validation_errors_by_type.get(error_type, 0) + 1
                            
                            if error_type == "unknown_condition":
                                self.unknown_condition_types.add(cond_type)
            
            problem_detail["hallucinations"] = hallucination_errors
            problem_detail["hallucination_count"] = hallucination_count
            self.hallucination_counts.append(hallucination_count)
            
            if recovery_steps:
                self.hallucination_recovery_steps.extend(recovery_steps)
        
        # Extract final validation info from result summary
        if "summary" in result and isinstance(result["summary"], dict):
            summary = result["summary"]
            # Additional summary info can be extracted here
        
        self.problem_details.append(problem_detail)
    
    def add_skipped_problem(self, problem_id: str, reason: str, error_types: List[str] = None):
        """
        Add a skipped problem (due to dataset error) to metrics.
        These problems are NOT counted in success/failure statistics.
        
        Args:
            problem_id: ID of the skipped problem
            reason: Reason for skipping (e.g., "Dataset validation error")
            error_types: List of error types that caused the skip
        """
        self.skipped_problems += 1
        self.skipped_problem_ids.append(problem_id)
        self.skipped_reasons[problem_id] = {
            "reason": reason,
            "error_types": error_types or []
        }
        
        # Track error types for analysis
        if error_types:
            for error_type in error_types:
                self.validation_errors_by_type[error_type] = \
                    self.validation_errors_by_type.get(error_type, 0) + 1
                if error_type == "unknown_condition":
                    # We don't have the specific condition type here, but it's logged elsewhere
                    pass
    
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error message into type."""
        error_lower = error_msg.lower()
        
        if any(x in error_lower for x in ['not defined', 'undefined', 'not found', 'unknown']):
            return "undefined_reference"
        elif any(x in error_lower for x in ['syntax error', 'parse error', 'invalid syntax']):
            return "syntax_error"
        elif any(x in error_lower for x in ['type error', 'type mismatch', 'expected', 'wrong type']):
            return "type_error"
        elif any(x in error_lower for x in ['invalid command', 'unknown command', 'no such command']):
            return "invalid_command"
        elif any(x in error_lower for x in ['duplicate', 'already defined', 'already exists']):
            return "duplicate_error"
        elif any(x in error_lower for x in ['constraint', 'cannot assert']):
            return "constraint_error"
        elif any(x in error_lower for x in ['output', 'mismatch', 'expected']):
            return "output_mismatch"
        else:
            return "other_error"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        # Calculate effective total (excluding skipped problems)
        effective_total = self.total_problems  # total_problems doesn't include skipped
        
        summary = {
            # Basic success metrics (skipped problems are EXCLUDED from success rate calculation)
            "total_problems": self.total_problems,  # Problems actually evaluated
            "successful_problems": self.successful_problems,
            "failed_problems": self.failed_problems,
            "skipped_problems": self.skipped_problems,  # Problems skipped due to dataset errors
            "success_rate": self.successful_problems / effective_total if effective_total > 0 else 0,
            
            # Skipped problems details
            "skipped_problem_ids": self.skipped_problem_ids,
            "skipped_reasons": self.skipped_reasons,
            
            # Step metrics
            "average_steps": sum(self.total_steps_per_problem) / len(self.total_steps_per_problem) if self.total_steps_per_problem else 0,
            "average_success_steps": sum(self.success_steps) / len(self.success_steps) if self.success_steps else 0,
            "min_success_steps": min(self.success_steps) if self.success_steps else 0,
            "max_success_steps": max(self.success_steps) if self.success_steps else 0,
            "success_step_distribution": self._get_step_distribution(),
            
            # Hallucination metrics
            "total_hallucinations": sum(self.hallucination_counts),
            "average_hallucinations_per_problem": sum(self.hallucination_counts) / len(self.hallucination_counts) if self.hallucination_counts else 0,
            "hallucination_error_types": self.hallucination_error_types,
            "average_hallucination_recovery_steps": sum(self.hallucination_recovery_steps) / len(self.hallucination_recovery_steps) if self.hallucination_recovery_steps else 0,
            
            # Object missing metrics
            "missing_objects_by_type": self.missing_objects_by_type,
            "total_missing_objects": sum(self.missing_objects_by_type.values()),
            
            # Condition failure metrics
            "failed_conditions_by_type": self.failed_conditions_by_type,
            "total_failed_conditions": sum(self.failed_conditions_by_type.values()),
            
            # Validation error metrics
            "validation_errors_count": len(self.validation_errors),
            "validation_errors_by_type": self.validation_errors_by_type,
            "unknown_condition_types": list(self.unknown_condition_types),
        }

        return summary
    
    def _get_step_distribution(self) -> Dict[int, int]:
        """Get distribution of success steps."""
        distribution = {}
        for step in self.success_steps:
            distribution[step] = distribution.get(step, 0) + 1
        return dict(sorted(distribution.items()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "summary": self.get_summary(),
            "problem_details": self.problem_details,
            "skipped_problems": {
                "count": self.skipped_problems,
                "ids": self.skipped_problem_ids,
                "reasons": self.skipped_reasons
            }
        }


class AgentBenchmarkRunner:
    """Run and evaluate agent on benchmark problems."""
    
    def __init__(self, model: str = "gpt-4o", max_iterations: int = 10,
                 save_images: bool = True, verbose: bool = False,
                 use_vision: bool = True, run_id: Optional[str] = None):
        """
        Initialize benchmark runner.
        
        Args:
            model: LLM model to use
            max_iterations: Max iterations per problem
            save_images: Save intermediate images
            verbose: Print detailed logs
            use_vision: Whether to send rendered images to LLM
            run_id: Custom run identifier for logging
        """
        self.model = model
        self.max_iterations = max_iterations
        self.save_images = save_images
        self.verbose = verbose
        self.use_vision = use_vision
        
        # Generate run_id if not provided
        if run_id is None:
            model_safe = model.replace("/", "_").replace(":", "_")
            vision_suffix = "vision" if use_vision else "no_vision"
            run_id = f"{model_safe}_{vision_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_id = run_id
        
        self.agent = ReActAgent(
            model=model,
            max_iterations=max_iterations,
            save_images=save_images,
            log_dir="agent_logs",
            verbose=verbose,
            use_vision=use_vision,
            run_id=run_id
        )

        # Store run directory for reference
        self.run_dir = self.agent.run_dir

        self.metrics = DetailedMetrics()
        
        # Initialize validation error logger
        self.validation_error_logger = ValidationErrorLogger(
            log_dir=os.path.join(self.run_dir, "validation_errors")
        )
        # Set as global logger so DSLValidator can use it
        set_validation_error_logger(self.validation_error_logger)
    
    def run_single(self, problem_id: str, dataset_path: str) -> Dict[str, Any]:
        """
        Run agent on a single problem.
        
        Args:
            problem_id: Problem ID to solve
            dataset_path: Path to benchmark dataset
            
        Returns:
            Results dictionary
        """
        dataset = BenchmarkDataset(dataset_path)
        problem = dataset.get_problem(problem_id)
        
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found in dataset")
        
        print(f"\nSolving Problem: {problem.id}")
        print(f"Problem: {problem.subject[:100]}...")
        print(f"Model: {self.model}")
        print(f"Max Iterations: {self.max_iterations}")
        print("="*70)
        
        results = self.agent.solve(problem)
        
        # Load memory for detailed analysis
        memory_data = None
        if results.get("memory_path") and os.path.exists(results["memory_path"]):
            with open(results["memory_path"], 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
        
        # Collect metrics
        self.metrics.reset()
        self.metrics.add_problem_result(results, memory_data)
        
        print("\n" + "="*70)
        print("Results:")
        print(f"  Success: {'✓' if results['success'] else '✗'}")
        print(f"  Iterations: {results['iterations']}/{self.max_iterations}")
        if 'summary' in results:
            summary = results['summary']
            print(f"  Duration: {summary['duration_seconds']:.1f}s")
            print(f"  Successful Executions: {summary['successful_executions']}")
        print(f"\n  Session ID: {results.get('session_id', 'N/A')}")
        print(f"  Log File: {results.get('log_file', 'N/A')}")
        if results.get('images'):
            print(f"  Images saved: {len(results['images'])}")
            for img in results['images'][:3]:  # Show first 3
                print(f"    - {img}")
            if len(results['images']) > 3:
                print(f"    ... and {len(results['images']) - 3} more")
        print("="*70)
        
        return results
    
    def run_batch(self, dataset_path: str, limit: Optional[int] = None,
                  start_idx: int = 0, output_file: str = "agent_results.json") -> Dict[str, Any]:
        """
        Run agent on multiple problems.
        
        Args:
            dataset_path: Path to benchmark dataset
            limit: Maximum number of problems (None for all)
            start_idx: Starting index
            output_file: Output file for results
            
        Returns:
            Evaluation report
        """
        dataset = BenchmarkDataset(dataset_path)
        
        problems = list(dataset)
        problems = problems[start_idx:]
        if limit:
            problems = problems[:limit]
        
        vision_mode = "with_vision" if self.use_vision else "no_vision"
        
        print(f"\n{'='*70}")
        print(f"Agent Benchmark Evaluation")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"Model: {self.model}")
        print(f"Vision: {'✓ Enabled' if self.use_vision else '✗ Disabled'}")
        print(f"Problems: {len(problems)} (from index {start_idx})")
        print(f"Max Iterations: {self.max_iterations}")
        print("="*70 + "\n")
        
        self.metrics.reset()
        results = []
        
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Problem {problem.id}")
            print(f"Subject: {problem.subject[:80]}...")
            
            try:
                result = self.agent.solve(problem)
                
                # Check for dataset errors in validation result
                validation_result = result.get("validation_result")
                if validation_result and validation_result.get("has_dataset_error"):
                    error_types = validation_result.get("dataset_error_types", [])
                    print(f"  ⚠️  SKIPPED: Dataset error detected ({', '.join(error_types)})")
                    self.metrics.add_skipped_problem(
                        problem_id=problem.id,
                        reason="Dataset validation error",
                        error_types=error_types
                    )
                    # Still save the result for reference but mark as skipped
                    result["skipped"] = True
                    result["skip_reason"] = f"Dataset error: {', '.join(error_types)}"
                    results.append(result)
                    continue  # Skip to next problem
                
                # Load memory for detailed analysis
                memory_data = None
                if result.get("memory_path") and os.path.exists(result["memory_path"]):
                    try:
                        with open(result["memory_path"], 'r', encoding='utf-8') as f:
                            memory_data = json.load(f)
                    except Exception as e:
                        print(f"  Warning: Could not load memory: {e}")
                
                # Collect metrics
                self.metrics.add_problem_result(result, memory_data)
                results.append(result)
                
                if result['success']:
                    print(f"  ✓ SOLVED in {result['iterations']} iterations")
                else:
                    print(f"  ✗ FAILED after {result['iterations']} iterations")
                
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                error_result = {
                    "problem_id": problem.id,
                    "success": False,
                    "error": str(e),
                    "iterations": 0
                }
                results.append(error_result)
                self.metrics.add_problem_result(error_result, None)
        
        # Get detailed metrics summary
        metrics_summary = self.metrics.get_summary()
        
        # Create comprehensive report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "dataset": dataset_path,
                "max_iterations": self.max_iterations,
                "start_idx": start_idx,
                "problems_count": len(problems),
                "use_vision": self.use_vision,
                "vision_mode": vision_mode,
                "run_id": self.run_id,
                "run_dir": self.run_dir
            },
            "metrics": metrics_summary,
            "detailed_analysis": {
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
            },
            "results": results,
            "problem_details": self.metrics.problem_details
        }
        
        # Add validation error details from error logger
        validation_error_summary = self.validation_error_logger.get_summary()
        report["validation_errors"] = {
            "summary": validation_error_summary,
            "errors": [e.to_dict() for e in self.validation_error_logger.errors]
        }
        
        # Save validation error log file
        if self.validation_error_logger.errors:
            validation_log_file = self.validation_error_logger.save_to_file()
            report["validation_error_log_file"] = validation_log_file
            print(f"\n⚠️  Validation errors logged to: {validation_log_file}")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        self._print_summary(metrics_summary, output_file)
        
        return report
    
    def _print_summary(self, metrics: Dict[str, Any], output_file: str):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        print("\n📈 1. Success Rate")
        print("-"*40)
        print(f"  Total Evaluated: {metrics['total_problems']}")
        print(f"  Successful: {metrics['successful_problems']} ({metrics['success_rate']:.1%})")
        print(f"  Failed: {metrics['failed_problems']}")
        if metrics.get('skipped_problems', 0) > 0:
            print(f"  ⚠️  Skipped (Dataset Errors): {metrics['skipped_problems']}")
            print(f"  (Success rate excludes skipped problems)")
        
        print("\n📉 2. Step Analysis")
        print("-"*40)
        print(f"  Average Steps to Success: {metrics['average_success_steps']:.2f}")
        print(f"  Min Steps: {metrics['min_success_steps']}")
        print(f"  Max Steps: {metrics['max_success_steps']}")
        if metrics['success_step_distribution']:
            print("  Step Distribution:")
            for step, count in sorted(metrics['success_step_distribution'].items()):
                pct = count / metrics['successful_problems'] * 100 if metrics['successful_problems'] > 0 else 0
                bar = '█' * int(pct / 5)
                print(f"    Step {step}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print("\n🔥 3. Hallucination Analysis (DSL Errors)")
        print("-"*40)
        print(f"  Total Hallucinations: {metrics['total_hallucinations']}")
        print(f"  Average per Problem: {metrics['average_hallucinations_per_problem']:.2f}")
        print(f"  Average Recovery Steps: {metrics['average_hallucination_recovery_steps']:.2f}")
        if metrics['hallucination_error_types']:
            print("  Error Type Distribution:")
            total_errors = sum(metrics['hallucination_error_types'].values())
            for error_type, count in sorted(metrics['hallucination_error_types'].items(), key=lambda x: -x[1]):
                pct = count / total_errors * 100 if total_errors > 0 else 0
                print(f"    {error_type}: {count} ({pct:.1f}%)")
        
        print("\n📦 4. Missing Objects Analysis")
        print("-"*40)
        print(f"  Total Missing Objects: {metrics['total_missing_objects']}")
        if metrics['missing_objects_by_type']:
            print("  By Type:")
            for obj_type, count in metrics['missing_objects_by_type'].items():
                if count > 0:
                    print(f"    {obj_type}: {count}")
        
        print("\n❌ 5. Failed Conditions Analysis")
        print("-"*40)
        print(f"  Total Failed Conditions: {metrics['total_failed_conditions']}")
        if metrics['failed_conditions_by_type']:
            print("  By Type:")
            for cond_type, count in sorted(metrics['failed_conditions_by_type'].items(), key=lambda x: -x[1]):
                print(f"    {cond_type}: {count}")
        
        print("\n⚠️  6. Dataset Errors & Skipped Problems")
        print("-"*40)
        skipped = metrics.get('skipped_problems', 0)
        print(f"  Skipped Problems: {skipped}")
        if skipped > 0:
            print(f"  Skipped Problem IDs: {metrics.get('skipped_problem_ids', [])[:10]}")  # Show first 10
            if skipped > 10:
                print(f"    ... and {skipped - 10} more")
        print(f"  Total Validation Errors: {metrics['validation_errors_count']}")
        if metrics['validation_errors_by_type']:
            print("  By Error Type:")
            for err_type, count in sorted(metrics['validation_errors_by_type'].items(), key=lambda x: -x[1]):
                print(f"    {err_type}: {count}")
        if metrics['unknown_condition_types']:
            print("  Unknown Condition Types Found:")
            for cond_type in sorted(metrics['unknown_condition_types']):
                print(f"    - {cond_type}")
        
        print("\n" + "="*70)
        print(f"📁 Results saved to: {output_file}")
        print(f"📂 Logs saved to: {self.run_dir}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ReAct agent on geometry benchmark"
    )
    
    parser.add_argument(
        "--problem-id",
        type=str,
        help="Solve a single problem by ID"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run on multiple problems"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ground_truth/geoqa3_dataset.json",
        help="Path to benchmark dataset"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems (for batch mode)"
    )
    
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index (for batch mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum iterations per problem"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results_{model}_{timestamp}.json)"
    )
    
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save intermediate images"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (implies verbose, saves images)"
    )
    
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision - don't send rendered images to LLM (for comparison)"
    )
    
    args = parser.parse_args()
    
    # Debug mode settings
    if args.debug:
        args.verbose = True
        args.no_save_images = False
    
    # Generate default output filename
    if args.output is None:
        model_name = args.model.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{model_name}_{timestamp}.json"
    
    # Check for API key
    if "gpt" in args.model.lower() or "openai" in args.model.lower():
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in environment")
            print("Please set it in .env file or environment variables")
            return 1
    elif "claude" in args.model.lower() or "anthropic" in args.model.lower():
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not found in environment")
            print("Please set it in .env file or environment variables")
            return 1
    
    # Determine vision mode
    use_vision = not args.no_vision
    vision_mode = "with_vision" if use_vision else "no_vision"
    
    print(f"\n{'='*70}")
    print(f"Vision Mode: {'✓ Enabled' if use_vision else '✗ Disabled'}")
    print(f"{'='*70}\n")
    
    # Initialize runner
    runner = AgentBenchmarkRunner(
        model=args.model,
        max_iterations=args.max_iter,
        save_images=not args.no_save_images,
        verbose=args.verbose,
        use_vision=use_vision
    )
    
    # Run
    if args.problem_id:
        # Single problem mode
        results = runner.run_single(args.problem_id, args.dataset)
        
        # Save single result with metrics
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "problem_id": args.problem_id,
                "use_vision": use_vision,
                "vision_mode": vision_mode
            },
            "result": results,
            "metrics": runner.metrics.to_dict()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        return 0 if results['success'] else 1
    
    elif args.batch:
        # Batch mode
        runner.run_batch(
            args.dataset, 
            limit=args.limit, 
            start_idx=args.start_idx,
            output_file=args.output
        )
        return 0
    
    else:
        print("ERROR: Must specify --problem-id or --batch")
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
