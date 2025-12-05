#!/usr/bin/env python3
"""
Run Agent Benchmark
Main script for evaluating ReAct agent on geometry problems.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from benchmark_dataset import BenchmarkDataset
from react_agent import ReActAgent


class AgentBenchmarkRunner:
    """Run and evaluate agent on benchmark problems."""
    
    def __init__(self, model: str = "gpt-4o", max_iterations: int = 10,
                 save_images: bool = True, verbose: bool = False):
        """
        Initialize benchmark runner.
        
        Args:
            model: LLM model to use
            max_iterations: Max iterations per problem
            save_images: Save intermediate images
            verbose: Print detailed logs
        """
        self.model = model
        self.max_iterations = max_iterations
        self.save_images = save_images
        self.verbose = verbose
        
        self.agent = ReActAgent(
            model=model,
            max_iterations=max_iterations,
            save_images=save_images,
            log_dir="agent_logs",
            verbose=verbose
        )
    
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
                  output_file: str = "agent_results.json") -> Dict[str, Any]:
        """
        Run agent on multiple problems.
        
        Args:
            dataset_path: Path to benchmark dataset
            limit: Maximum number of problems (None for all)
            output_file: Output file for results
            
        Returns:
            Evaluation report
        """
        dataset = BenchmarkDataset(dataset_path)
        
        problems = list(dataset)
        if limit:
            problems = problems[:limit]
        
        print(f"\n{'='*70}")
        print(f"Agent Benchmark Evaluation")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"Model: {self.model}")
        print(f"Problems: {len(problems)}")
        print(f"Max Iterations: {self.max_iterations}")
        print("="*70 + "\n")
        
        results = []
        success_count = 0
        total_iterations = 0
        
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Problem {problem.id}")
            print(f"Subject: {problem.subject[:80]}...")
            
            try:
                result = self.agent.solve(problem)
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    print(f"  ✓ SOLVED in {result['iterations']} iterations")
                else:
                    print(f"  ✗ FAILED after {result['iterations']} iterations")
                
                total_iterations += result['iterations']
                
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                results.append({
                    "problem_id": problem.id,
                    "success": False,
                    "error": str(e),
                    "iterations": 0
                })
        
        # Calculate metrics
        success_rate = success_count / len(problems) if problems else 0
        avg_iterations = total_iterations / len(problems) if problems else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "dataset": dataset_path,
            "total_problems": len(problems),
            "successful": success_count,
            "failed": len(problems) - success_count,
            "success_rate": success_rate,
            "average_iterations": avg_iterations,
            "max_iterations": self.max_iterations,
            "results": results
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Problems: {len(problems)}")
        print(f"Successful: {success_count} ({success_rate:.1%})")
        print(f"Failed: {len(problems) - success_count}")
        print(f"Average Iterations: {avg_iterations:.1f}")
        print(f"Results saved to: {output_file}")
        print("="*70)
        
        return report


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
        default="benchmark_geoqa3.json",
        help="Path to benchmark dataset"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems (for batch mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        # choices=["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "claude-3-5-sonnet-20241022"],
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
        default="agent_results.json",
        help="Output file for results"
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
    
    args = parser.parse_args()
    
    # Debug mode settings
    if args.debug:
        args.verbose = True
        args.no_save_images = False
    
    # Check for API key
    if "gpt" in args.model.lower():
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in environment")
            print("Please set it in .env file or environment variables")
            return 1
    elif "claude" in args.model.lower():
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not found in environment")
            print("Please set it in .env file or environment variables")
            return 1
    
    # Initialize runner
    runner = AgentBenchmarkRunner(
        model=args.model,
        max_iterations=args.max_iter,
        save_images=not args.no_save_images,
        verbose=args.verbose
    )
    
    # Run
    if args.problem_id:
        # Single problem mode
        results = runner.run_single(args.problem_id, args.dataset)
        
        # Save single result
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        return 0 if results['success'] else 1
    
    elif args.batch:
        # Batch mode
        runner.run_batch(args.dataset, limit=args.limit, output_file=args.output)
        return 0
    
    else:
        print("ERROR: Must specify --problem-id or --batch")
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

