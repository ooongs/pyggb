#!/usr/bin/env python3
"""
Analyze Benchmark Results
Comprehensive analysis tool for comparing multiple model results.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import glob


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results from multiple models."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.model_names: List[str] = []
    
    def load_result(self, filepath: str) -> bool:
        """Load a single result file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract model name and vision mode from metadata
            metadata = data.get("metadata", {})
            model_name = metadata.get("model", "unknown")
            
            # Include vision mode in model name for comparison
            use_vision = metadata.get("use_vision", True)
            vision_mode = metadata.get("vision_mode", "with_vision" if use_vision else "no_vision")
            
            # Create display name including vision mode
            display_name = f"{model_name} ({vision_mode})"
            
            self.results.append(data)
            self.model_names.append(display_name)
            return True
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False
    
    def load_results_from_dir(self, directory: str, pattern: str = "results_*.json"):
        """Load all result files from a directory."""
        files = glob.glob(os.path.join(directory, pattern))
        for filepath in sorted(files):
            self.load_result(filepath)
        print(f"Loaded {len(self.results)} result files")
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare metrics across all loaded models."""
        if not self.results:
            return {}
        
        comparison = {
            "models": [],
            "rankings": {},
            "detailed_comparison": {}
        }
        
        model_metrics = []
        
        for i, (result, model_name) in enumerate(zip(self.results, self.model_names)):
            metrics = result.get("metrics", {})
            metadata = result.get("metadata", {})
            
            model_data = {
                "model": model_name,
                "base_model": metadata.get("model", "unknown"),
                "use_vision": metadata.get("use_vision", True),
                "vision_mode": metadata.get("vision_mode", "with_vision"),
                "success_rate": metrics.get("success_rate", 0),
                "total_problems": metrics.get("total_problems", 0),
                "successful_problems": metrics.get("successful_problems", 0),
                "average_success_steps": metrics.get("average_success_steps", 0),
                "total_hallucinations": metrics.get("total_hallucinations", 0),
                "average_hallucinations_per_problem": metrics.get("average_hallucinations_per_problem", 0),
                "average_hallucination_recovery_steps": metrics.get("average_hallucination_recovery_steps", 0),
                "total_missing_objects": metrics.get("total_missing_objects", 0),
                "total_failed_conditions": metrics.get("total_failed_conditions", 0),
                "hallucination_error_types": metrics.get("hallucination_error_types", {}),
                "missing_objects_by_type": metrics.get("missing_objects_by_type", {}),
                "failed_conditions_by_type": metrics.get("failed_conditions_by_type", {})
            }
            model_metrics.append(model_data)
        
        comparison["models"] = model_metrics
        
        # Calculate rankings
        if len(model_metrics) > 1:
            # Rank by success rate (higher is better)
            comparison["rankings"]["success_rate"] = sorted(
                [(m["model"], m["success_rate"]) for m in model_metrics],
                key=lambda x: -x[1]
            )
            
            # Rank by average steps to success (lower is better)
            comparison["rankings"]["efficiency"] = sorted(
                [(m["model"], m["average_success_steps"]) for m in model_metrics if m["successful_problems"] > 0],
                key=lambda x: x[1]
            )
            
            # Rank by hallucination rate (lower is better)
            comparison["rankings"]["hallucination_rate"] = sorted(
                [(m["model"], m["average_hallucinations_per_problem"]) for m in model_metrics],
                key=lambda x: x[1]
            )
            
            # Rank by object accuracy (lower missing is better)
            comparison["rankings"]["object_accuracy"] = sorted(
                [(m["model"], m["total_missing_objects"]) for m in model_metrics],
                key=lambda x: x[1]
            )
            
            # Rank by condition accuracy (lower failed is better)
            comparison["rankings"]["condition_accuracy"] = sorted(
                [(m["model"], m["total_failed_conditions"]) for m in model_metrics],
                key=lambda x: x[1]
            )
        
        return comparison
    
    def generate_report(self, output_format: str = "text") -> str:
        """Generate comparison report."""
        comparison = self.compare_models()
        
        if output_format == "json":
            return json.dumps(comparison, indent=2, ensure_ascii=False)
        else:
            return self._format_text_report(comparison)
    
    def _format_text_report(self, comparison: Dict[str, Any]) -> str:
        """Format comparison as text report."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("📊 BENCHMARK RESULTS COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        if not comparison.get("models"):
            lines.append("\nNo results to compare.")
            return "\n".join(lines)
        
        models = comparison["models"]
        
        # 1. Success Rate Comparison
        lines.append("\n" + "─" * 80)
        lines.append("📈 1. SUCCESS RATE COMPARISON")
        lines.append("─" * 80)
        lines.append(f"{'Model':<40} {'Success Rate':<15} {'Solved/Total':<15}")
        lines.append("-" * 70)
        for m in sorted(models, key=lambda x: -x["success_rate"]):
            rate = f"{m['success_rate']:.1%}"
            solved = f"{m['successful_problems']}/{m['total_problems']}"
            lines.append(f"{m['model']:<40} {rate:<15} {solved:<15}")
        
        # 2. Step Analysis
        lines.append("\n" + "─" * 80)
        lines.append("⏱️ 2. STEP ANALYSIS (Efficiency)")
        lines.append("─" * 80)
        lines.append(f"{'Model':<40} {'Avg Steps':<15} {'Description':<25}")
        lines.append("-" * 80)
        for m in sorted(models, key=lambda x: x["average_success_steps"] if x["successful_problems"] > 0 else float('inf')):
            if m["successful_problems"] > 0:
                avg = f"{m['average_success_steps']:.2f}"
                desc = self._get_efficiency_description(m["average_success_steps"])
            else:
                avg = "N/A"
                desc = "No successful solutions"
            lines.append(f"{m['model']:<40} {avg:<15} {desc:<25}")
        
        # 3. Hallucination Analysis (DSL Errors)
        lines.append("\n" + "─" * 80)
        lines.append("🔥 3. HALLUCINATION ANALYSIS (DSL Execution Errors)")
        lines.append("─" * 80)
        lines.append(f"{'Model':<40} {'Total':<10} {'Avg/Problem':<15} {'Avg Recovery':<15}")
        lines.append("-" * 80)
        for m in sorted(models, key=lambda x: x["average_hallucinations_per_problem"]):
            total = str(m["total_hallucinations"])
            avg = f"{m['average_hallucinations_per_problem']:.2f}"
            recovery = f"{m['average_hallucination_recovery_steps']:.2f}" if m["average_hallucination_recovery_steps"] > 0 else "N/A"
            lines.append(f"{m['model']:<40} {total:<10} {avg:<15} {recovery:<15}")
        
        # Error type breakdown per model
        lines.append("\n  Error Type Breakdown:")
        for m in models:
            if m["hallucination_error_types"]:
                lines.append(f"\n  [{m['model']}]")
                for error_type, count in sorted(m["hallucination_error_types"].items(), key=lambda x: -x[1]):
                    lines.append(f"    • {error_type}: {count}")
        
        # 4. Object Accuracy Analysis
        lines.append("\n" + "─" * 80)
        lines.append("📦 4. OBJECT ACCURACY (Missing Objects)")
        lines.append("─" * 80)
        lines.append(f"{'Model':<40} {'Total Missing':<15} {'Points':<10} {'Segments':<10} {'Lines':<10}")
        lines.append("-" * 85)
        for m in sorted(models, key=lambda x: x["total_missing_objects"]):
            total = str(m["total_missing_objects"])
            obj = m.get("missing_objects_by_type", {})
            points = str(obj.get("points", 0))
            segments = str(obj.get("segments", 0))
            lines_obj = str(obj.get("lines", 0))
            lines.append(f"{m['model']:<40} {total:<15} {points:<10} {segments:<10} {lines_obj:<10}")
        
        # 5. Condition Accuracy Analysis
        lines.append("\n" + "─" * 80)
        lines.append("❌ 5. CONDITION ACCURACY (Failed Conditions)")
        lines.append("─" * 80)
        lines.append(f"{'Model':<40} {'Total Failed':<15}")
        lines.append("-" * 55)
        for m in sorted(models, key=lambda x: x["total_failed_conditions"]):
            total = str(m["total_failed_conditions"])
            lines.append(f"{m['model']:<40} {total:<15}")
        
        # Failed condition type breakdown
        lines.append("\n  Failed Condition Type Breakdown:")
        for m in models:
            if m["failed_conditions_by_type"]:
                lines.append(f"\n  [{m['model']}]")
                for cond_type, count in sorted(m["failed_conditions_by_type"].items(), key=lambda x: -x[1]):
                    lines.append(f"    • {cond_type}: {count}")
        
        # 6. Rankings Summary
        if "rankings" in comparison and comparison["rankings"]:
            lines.append("\n" + "─" * 80)
            lines.append("🏆 6. OVERALL RANKINGS")
            lines.append("─" * 80)
            
            rankings = comparison["rankings"]
            
            if "success_rate" in rankings:
                lines.append("\n  🥇 Best Success Rate:")
                for i, (model, rate) in enumerate(rankings["success_rate"][:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    lines.append(f"    {medal} {model}: {rate:.1%}")
            
            if "efficiency" in rankings and rankings["efficiency"]:
                lines.append("\n  ⚡ Most Efficient (Fewest Steps):")
                for i, (model, steps) in enumerate(rankings["efficiency"][:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    lines.append(f"    {medal} {model}: {steps:.2f} avg steps")
            
            if "hallucination_rate" in rankings:
                lines.append("\n  🎯 Lowest Hallucination Rate:")
                for i, (model, rate) in enumerate(rankings["hallucination_rate"][:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    lines.append(f"    {medal} {model}: {rate:.2f} avg errors/problem")
            
            if "object_accuracy" in rankings:
                lines.append("\n  📦 Best Object Accuracy (Fewest Missing):")
                for i, (model, count) in enumerate(rankings["object_accuracy"][:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    lines.append(f"    {medal} {model}: {count} missing objects")
            
            if "condition_accuracy" in rankings:
                lines.append("\n  ✅ Best Condition Accuracy (Fewest Failed):")
                for i, (model, count) in enumerate(rankings["condition_accuracy"][:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    lines.append(f"    {medal} {model}: {count} failed conditions")
        
        # 7. Key Insights
        lines.append("\n" + "─" * 80)
        lines.append("💡 7. KEY INSIGHTS")
        lines.append("─" * 80)
        
        if len(models) > 1:
            best_success = max(models, key=lambda x: x["success_rate"])
            worst_success = min(models, key=lambda x: x["success_rate"])
            
            lines.append(f"\n  • Best performing model: {best_success['model']} ({best_success['success_rate']:.1%} success)")
            lines.append(f"  • Worst performing model: {worst_success['model']} ({worst_success['success_rate']:.1%} success)")
            
            if best_success["success_rate"] > 0:
                diff = best_success["success_rate"] - worst_success["success_rate"]
                lines.append(f"  • Performance gap: {diff:.1%}")
            
            # Average metrics across all models
            avg_hallucination = sum(m["average_hallucinations_per_problem"] for m in models) / len(models)
            lines.append(f"\n  • Average hallucination rate across models: {avg_hallucination:.2f} per problem")
            
            total_missing = sum(m["total_missing_objects"] for m in models)
            lines.append(f"  • Total missing objects across all models: {total_missing}")
            
            total_failed = sum(m["total_failed_conditions"] for m in models)
            lines.append(f"  • Total failed conditions across all models: {total_failed}")
        else:
            m = models[0]
            lines.append(f"\n  Single model analysis: {m['model']}")
            lines.append(f"  • Success rate: {m['success_rate']:.1%}")
            lines.append(f"  • Average steps to success: {m['average_success_steps']:.2f}")
            lines.append(f"  • Hallucination rate: {m['average_hallucinations_per_problem']:.2f} per problem")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def _get_efficiency_description(self, avg_steps: float) -> str:
        """Get efficiency description based on average steps."""
        if avg_steps <= 1:
            return "Excellent (1st try)"
        elif avg_steps <= 2:
            return "Very Good"
        elif avg_steps <= 3:
            return "Good"
        elif avg_steps <= 5:
            return "Average"
        else:
            return "Needs Improvement"
    
    def generate_csv_summary(self) -> str:
        """Generate CSV format summary for easy import to spreadsheet."""
        comparison = self.compare_models()
        
        if not comparison.get("models"):
            return "No data"
        
        headers = [
            "Model",
            "Success_Rate",
            "Successful",
            "Total",
            "Avg_Steps_to_Success",
            "Total_Hallucinations",
            "Avg_Hallucinations_per_Problem",
            "Avg_Recovery_Steps",
            "Total_Missing_Objects",
            "Missing_Points",
            "Missing_Segments",
            "Missing_Lines",
            "Total_Failed_Conditions"
        ]
        
        lines = [",".join(headers)]
        
        for m in comparison["models"]:
            obj = m.get("missing_objects_by_type", {})
            row = [
                m["model"],
                f"{m['success_rate']:.4f}",
                str(m["successful_problems"]),
                str(m["total_problems"]),
                f"{m['average_success_steps']:.2f}",
                str(m["total_hallucinations"]),
                f"{m['average_hallucinations_per_problem']:.2f}",
                f"{m['average_hallucination_recovery_steps']:.2f}",
                str(m["total_missing_objects"]),
                str(obj.get("points", 0)),
                str(obj.get("segments", 0)),
                str(obj.get("lines", 0)),
                str(m["total_failed_conditions"])
            ]
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    def generate_per_problem_analysis(self) -> str:
        """Generate per-problem analysis comparing models."""
        if len(self.results) < 2:
            return "Need at least 2 models for comparison"
        
        lines = []
        lines.append("=" * 80)
        lines.append("📋 PER-PROBLEM ANALYSIS")
        lines.append("=" * 80)
        
        # Collect problem results by problem_id
        problem_results = {}
        
        for result, model_name in zip(self.results, self.model_names):
            for prob in result.get("problem_details", []):
                pid = prob.get("problem_id")
                if pid not in problem_results:
                    problem_results[pid] = {}
                problem_results[pid][model_name] = prob
        
        # Compare
        lines.append(f"\n{'Problem ID':<15} " + " ".join(f"{m:<20}" for m in self.model_names))
        lines.append("-" * (15 + 21 * len(self.model_names)))
        
        for pid in sorted(problem_results.keys()):
            row = [f"{pid:<15}"]
            for model_name in self.model_names:
                if model_name in problem_results[pid]:
                    prob = problem_results[pid][model_name]
                    success = "✓" if prob.get("success") else "✗"
                    steps = prob.get("iterations", 0) if prob.get("success") else "-"
                    hall = prob.get("hallucination_count", 0)
                    row.append(f"{success} S:{steps} H:{hall}".ljust(20))
                else:
                    row.append("-".ljust(20))
            lines.append(" ".join(row))
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare benchmark results"
    )
    
    parser.add_argument(
        "result_files",
        nargs="*",
        help="Result JSON files to analyze"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing result files"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="results_*.json",
        help="File pattern for directory search"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: stdout)"
    )
    
    parser.add_argument(
        "--per-problem",
        action="store_true",
        help="Include per-problem comparison"
    )
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer()
    
    # Load results
    if args.dir:
        analyzer.load_results_from_dir(args.dir, args.pattern)
    elif args.result_files:
        for filepath in args.result_files:
            analyzer.load_result(filepath)
    else:
        print("ERROR: Must specify result files or --dir")
        parser.print_help()
        return 1
    
    if not analyzer.results:
        print("ERROR: No results loaded")
        return 1
    
    # Generate report
    if args.format == "csv":
        report = analyzer.generate_csv_summary()
    elif args.format == "json":
        report = analyzer.generate_report("json")
    else:
        report = analyzer.generate_report("text")
        if args.per_problem:
            report += "\n\n" + analyzer.generate_per_problem_analysis()
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


