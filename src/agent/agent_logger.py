#!/usr/bin/env python3
"""
Agent Logger
Comprehensive logging for ReAct agent sessions.
Organizes logs by timestamp folder for easy access.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import shutil

from src.utils import get_output_dir


class AgentLogger:
    """Logger for agent execution with detailed output saving.
    
    New folder structure:
    agent_logs/
    └── 20251207_161413/  (run timestamp folder)
        ├── images/
        │   ├── problem_id_iter1.png
        │   └── ...
        ├── dsl/
        │   ├── problem_id_iter1.txt
        │   └── ...
        ├── sessions/
        │   ├── problem_id.log
        │   └── problem_id_summary.json
        └── run_info.json
    """
    
    def __init__(self, log_dir: Optional[str] = None, save_images: bool = True,
                 run_id: Optional[str] = None, verbose: bool = False,
                 resume_mode: bool = False):
        """
        Initialize agent logger.

        Args:
            log_dir: Base directory for logs (default: agent_logs in project root)
            save_images: Whether to save images
            run_id: Custom run identifier (default: timestamp)
            verbose: Print verbose logging
            resume_mode: Whether this is a resume run (merge results instead of overwrite)
        """
        self.base_log_dir = log_dir or str(get_output_dir("agent_logs"))
        self.save_images = save_images
        self.verbose = verbose
        self.resume_mode = resume_mode
        self.session_start = datetime.now()

        # Create run-specific folder with timestamp
        if run_id is None:
            run_id = self.session_start.strftime('%Y%m%d_%H%M%S')
        self.run_id = run_id

        # Set up run directory structure
        self.run_dir = os.path.join(log_dir, run_id)
        self.images_dir = os.path.join(self.run_dir, "images")
        self.dsl_dir = os.path.join(self.run_dir, "dsl")
        self.sessions_dir = os.path.join(self.run_dir, "sessions")

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.dsl_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)

        # Track problems in this run
        self.problems_logged = []

        # Resume mode: Load existing run_info.json to merge results
        if self.resume_mode:
            existing_info_file = os.path.join(self.run_dir, "run_info.json")
            if os.path.exists(existing_info_file):
                try:
                    with open(existing_info_file, 'r', encoding='utf-8') as f:
                        existing_info = json.load(f)
                        self.problems_logged = existing_info.get("problems", [])
                    if self.verbose:
                        print(f"📂 Loaded existing run_info.json with {len(self.problems_logged)} problems")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load existing run_info.json: {e}")
                    print(f"   Starting with empty problems list")
            else:
                print(f"⚠️  Resume mode but run_info.json not found at {existing_info_file}")
                print(f"   Starting with empty problems list")

        # Save run info
        self._save_run_info()
    
    def _backup_file(self, file_path: str):
        """Create timestamped backup of existing file.

        Args:
            file_path: Path to file to backup
        """
        if not os.path.exists(file_path):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        backup_name = f"{name_without_ext}_backup_{timestamp}.json"
        backup_path = os.path.join(os.path.dirname(file_path), backup_name)

        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"📦 Backed up to: {backup_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")

    def _save_run_info(self):
        """Save run information to JSON file with backup in resume mode."""
        run_info = {
            "run_id": self.run_id,
            "start_time": self.session_start.isoformat(),
            "log_dir": self.run_dir,
            "save_images": self.save_images,
            "problems": self.problems_logged
        }

        info_file = os.path.join(self.run_dir, "run_info.json")

        # Resume mode: Create backup before overwriting
        if self.resume_mode and os.path.exists(info_file):
            self._backup_file(info_file)

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)
    
    def start_problem(self, problem_id: str, problem_text: str) -> str:
        """
        Start logging a new problem.

        Args:
            problem_id: Problem identifier
            problem_text: Problem description

        Returns:
            Session ID (format: problem_id_timestamp)
        """
        session_id = f"{problem_id}_{self.run_id}"

        # In resume mode, remove existing entry for this problem_id to avoid duplicates
        if self.resume_mode:
            self.problems_logged = [
                p for p in self.problems_logged
                if p.get("problem_id") != problem_id
            ]

        # Track this problem
        self.problems_logged.append({
            "problem_id": problem_id,
            "session_id": session_id,
            "start_time": datetime.now().isoformat()
        })
        self._save_run_info()
        
        # Create problem log file
        log_file = os.path.join(self.sessions_dir, f"{problem_id}.log")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"REACT AGENT SESSION\n")
            f.write("="*80 + "\n")
            f.write(f"Problem ID: {problem_id}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"Problem: {problem_text}\n")
            f.write("="*80 + "\n\n")
        
        return session_id
    
    def _get_problem_id_from_session(self, session_id: str) -> str:
        """Extract problem_id from session_id."""
        # session_id format: problem_id_timestamp
        # Need to handle problem_id that might contain underscores
        parts = session_id.rsplit('_', 2)
        if len(parts) >= 3:
            return parts[0]
        return session_id.split('_')[0]
    
    def log_iteration(self, session_id: str, iteration: int, 
                     thought: str, action_type: str, dsl_code: str,
                     success: bool, error: Optional[str] = None,
                     image_path: Optional[str] = None,
                     validation_result: Optional[Dict] = None):
        """
        Log a single ReAct iteration.
        
        Args:
            session_id: Session identifier
            iteration: Iteration number
            thought: Agent's thought
            action_type: Type of action
            dsl_code: Generated DSL code
            success: Whether execution succeeded
            error: Error message if failed
            image_path: Path to generated image
            validation_result: Validation results if available
        """
        problem_id = self._get_problem_id_from_session(session_id)
        log_file = os.path.join(self.sessions_dir, f"{problem_id}.log")
        
        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ITERATION {iteration}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n\n")
            
            f.write(f"THOUGHT:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{thought}\n\n")
            
            f.write(f"ACTION: {action_type}\n")
            f.write(f"{'-'*80}\n")
            
            if action_type in ['generate_dsl', 'modify_dsl']:
                f.write(f"DSL CODE:\n")
                f.write(f"```\n")
                f.write(f"{dsl_code}\n")
                f.write(f"```\n\n")
                
                # Save DSL code to separate file
                dsl_file = os.path.join(
                    self.dsl_dir, 
                    f"{problem_id}_iter{iteration}.txt"
                )
                with open(dsl_file, 'w', encoding='utf-8') as df:
                    df.write(dsl_code)
                
                f.write(f"EXECUTION RESULT:\n")
                f.write(f"{'-'*80}\n")
                if success:
                    f.write(f"✓ SUCCESS\n")
                    if image_path:
                        f.write(f"Image saved: {image_path}\n")
                        # Copy image to run directory
                        if self.save_images and os.path.exists(image_path):
                            dest_path = os.path.join(
                                self.images_dir,
                                f"{problem_id}_iter{iteration}.png"
                            )
                            shutil.copy2(image_path, dest_path)
                            f.write(f"Image copied to: {dest_path}\n")
                else:
                    f.write(f"✗ FAILED\n")
                    if error:
                        f.write(f"Error Details:\n")
                        f.write(f"{error}\n")
                
                # Log validation results if available
                if validation_result:
                    self._write_validation_result(f, validation_result)
            else:
                f.write(f"{dsl_code}\n")
            
            f.write(f"\n")
    
    def _write_validation_result(self, f, validation_result: Dict):
        """Write validation results to log file."""
        f.write(f"\nVALIDATION RESULTS:\n")
        f.write(f"{'-'*80}\n")

        # Check for validation error first
        if validation_result.get('error_message'):
            f.write(f"✗ VALIDATION ERROR:\n")
            f.write(f"{validation_result['error_message']}\n")
            f.write(f"\n")
            return

        # Normal validation results
        f.write(f"Object Score:    {validation_result.get('object_score', 0):.1%}\n")
        f.write(f"Condition Score: {validation_result.get('condition_score', 0):.1%}\n")
        f.write(f"Total Score:     {validation_result.get('total_score', 0):.1%}\n")
        f.write(f"Success:         {'✓ YES' if validation_result.get('success', False) else '✗ NO'}\n")

        # Details section with found objects
        if validation_result.get('details'):
            details = validation_result['details']

            # Show found objects
            found_objects = details.get('found_objects', {})
            has_found = any(objs for objs in found_objects.values() if objs)

            if has_found:
                f.write(f"\n📦 Found Objects:\n")
                for obj_type, objs in found_objects.items():
                    if objs:
                        objs_str = ', '.join(str(o) for o in objs) if isinstance(objs, list) else str(objs)
                        f.write(f"  • {obj_type.capitalize()}: {objs_str}\n")

        # Missing objects
        missing_objects = validation_result.get('missing_objects', {})
        has_missing = any(objs for objs in missing_objects.values() if objs)
        if has_missing:
            f.write(f"\n❌ Missing Objects:\n")
            for obj_type, objs in missing_objects.items():
                if objs:
                    objs_str = ', '.join(str(o) for o in objs) if isinstance(objs, list) else str(objs)
                    f.write(f"  • {obj_type.capitalize()}: {objs_str}\n")

        # Passed and Failed conditions with detailed messages
        cond_details = validation_result.get('details', {}).get('condition_details', [])
        failed_conditions = validation_result.get('failed_conditions', [])

        # Show passed conditions
        passed_details = [d for d in cond_details if d.get('passed', False)]
        if passed_details:
            f.write(f"\n✅ Passed Conditions ({len(passed_details)} total):\n")
            for i, detail in enumerate(passed_details, 1):
                cond = detail.get('condition', {})
                message = detail.get('message', 'No message')
                cond_type = cond.get('type', 'unknown')
                f.write(f"  {i}. [{cond_type}] {message}\n")

        # Show failed conditions with enhanced details
        failed_details = [d for d in cond_details if not d.get('passed', False)]
        if failed_details:
            f.write(f"\n❌ Failed Conditions ({len(failed_details)} total):\n")
            for i, detail in enumerate(failed_details, 1):
                cond = detail.get('condition', {})
                message = detail.get('message', 'No message')
                cond_type = cond.get('type', 'unknown')
                f.write(f"  {i}. [{cond_type}] {message}\n")

                # Show additional parameters for context
                shown_keys = {'type'}  # Already shown above
                for key, value in cond.items():
                    if key not in shown_keys and value is not None:
                        # Format the value nicely
                        if isinstance(value, (list, tuple)):
                            value_str = ', '.join(str(v) for v in value)
                        elif isinstance(value, float):
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                        f.write(f"     → {key}: {value_str}\n")
        elif failed_conditions:
            # If we have failed_conditions but no cond_details, show them anyway
            f.write(f"\n❌ Failed Conditions ({len(failed_conditions)} total):\n")
            for i, cond in enumerate(failed_conditions, 1):
                cond_type = cond.get('type', 'unknown')
                message = cond.get('validation_message', 'No message')
                f.write(f"  {i}. [{cond_type}] {message}\n")

                # Show additional parameters
                shown_keys = {'type', 'validation_message'}
                for key, value in cond.items():
                    if key not in shown_keys and value is not None:
                        if isinstance(value, (list, tuple)):
                            value_str = ', '.join(str(v) for v in value)
                        elif isinstance(value, float):
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                        f.write(f"     → {key}: {value_str}\n")

        # Show success message if all passed
        if not has_missing and not failed_details and not failed_conditions:
            if validation_result.get('total_score', 0) >= 0.9:
                f.write(f"\n✅ All objects and conditions satisfied!\n")

        # Warning if scores are low but no specific errors
        if validation_result.get('total_score', 0) < 0.5 and not has_missing and not failed_details and not failed_conditions:
            f.write(f"\n⚠️  Low score but no detailed error information available.\n")
            f.write(f"   This may indicate a validation bug or dataset error.\n")
    
    def log_final_result(self, session_id: str, success: bool, 
                        iterations: int, validation_result: Optional[Dict] = None):
        """
        Log final result of the session.
        
        Args:
            session_id: Session identifier
            success: Whether problem was solved
            iterations: Total iterations
            validation_result: Validation details
        """
        problem_id = self._get_problem_id_from_session(session_id)
        log_file = os.path.join(self.sessions_dir, f"{problem_id}.log")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL RESULT\n")
            f.write(f"{'='*80}\n")
            f.write(f"End Time: {datetime.now().isoformat()}\n")
            f.write(f"Total Iterations: {iterations}\n")
            f.write(f"Success: {'✓ YES' if success else '✗ NO'}\n")
            
            if validation_result:
                self._write_validation_result(f, validation_result)
            
            f.write(f"\n{'='*80}\n")
        
        # Update run info
        for p in self.problems_logged:
            if p["session_id"] == session_id:
                p["success"] = success
                p["iterations"] = iterations
                p["end_time"] = datetime.now().isoformat()
        self._save_run_info()
    
    def create_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Create a summary JSON for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dictionary
        """
        problem_id = self._get_problem_id_from_session(session_id)
        log_file = os.path.join(self.sessions_dir, f"{problem_id}.log")
        
        if not os.path.exists(log_file):
            return {}
        
        # Count iterations and collect files
        dsl_files = [
            f for f in os.listdir(self.dsl_dir)
            if f.startswith(f"{problem_id}_iter")
        ] if os.path.exists(self.dsl_dir) else []
        
        image_files = []
        if self.save_images and os.path.exists(self.images_dir):
            image_files = [
                f for f in os.listdir(self.images_dir)
                if f.startswith(f"{problem_id}_iter")
            ]
        
        summary = {
            "session_id": session_id,
            "problem_id": problem_id,
            "run_id": self.run_id,
            "log_file": log_file,
            "iterations": len(dsl_files),
            "dsl_files": [os.path.join(self.dsl_dir, f) for f in sorted(dsl_files)],
            "images": [os.path.join(self.images_dir, f) for f in sorted(image_files)],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary JSON
        summary_file = os.path.join(self.sessions_dir, f"{problem_id}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def get_all_sessions(self) -> list:
        """Get list of all logged sessions in this run."""
        if not os.path.exists(self.sessions_dir):
            return []
        
        log_files = [f for f in os.listdir(self.sessions_dir) if f.endswith('.log')]
        return sorted([f.replace('.log', '') for f in log_files])
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of the current run."""
        total_problems = len(self.problems_logged)
        successful = sum(1 for p in self.problems_logged if p.get("success", False))
        
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "start_time": self.session_start.isoformat(),
            "total_problems": total_problems,
            "successful": successful,
            "failed": total_problems - successful,
            "success_rate": successful / total_problems if total_problems > 0 else 0,
            "problems": self.problems_logged
        }


# Test function
if __name__ == "__main__":
    print("="*70)
    print("Agent Logger Test (New Structure)")
    print("="*70)
    print()
    
    # Create logger
    logger = AgentLogger(log_dir="test_agent_logs")
    
    print(f"Run directory: {logger.run_dir}")
    print()
    
    # Start problem
    session_id = logger.start_problem(
        "test_1",
        "Create a triangle ABC with specific angles"
    )
    print(f"Started session: {session_id}")
    
    # Log iteration
    logger.log_iteration(
        session_id,
        iteration=1,
        thought="I need to create three points for the triangle",
        action_type="generate_dsl",
        dsl_code="point :  -> A\npoint :  -> B\npoint :  -> C\npolygon : A B C -> triangle c a b",
        success=True,
        image_path=None
    )
    print(f"Logged iteration 1")
    
    # Log final result
    logger.log_final_result(
        session_id,
        success=True,
        iterations=1,
        validation_result={
            "object_score": 0.9,
            "condition_score": 0.8,
            "total_score": 0.85
        }
    )
    print(f"Logged final result")
    
    # Create summary
    summary = logger.create_session_summary(session_id)
    print(f"\nSession Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get run summary
    run_summary = logger.get_run_summary()
    print(f"\nRun Summary:")
    print(json.dumps(run_summary, indent=2))
    
    print(f"\n✓ Logger test complete!")
    print(f"Check logs in: {logger.run_dir}")
