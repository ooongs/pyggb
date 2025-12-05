#!/usr/bin/env python3
"""
Agent Logger
Comprehensive logging for ReAct agent sessions.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import shutil


class AgentLogger:
    """Logger for agent execution with detailed output saving."""
    
    def __init__(self, log_dir: str = "agent_logs", save_images: bool = True):
        """
        Initialize agent logger.
        
        Args:
            log_dir: Directory to save logs
            save_images: Whether to save images
        """
        self.log_dir = log_dir
        self.save_images = save_images
        self.session_start = datetime.now()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create subdirectories
        if save_images:
            os.makedirs(os.path.join(log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "dsl_code"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "sessions"), exist_ok=True)
    
    def start_problem(self, problem_id: str, problem_text: str) -> str:
        """
        Start logging a new problem.
        
        Args:
            problem_id: Problem identifier
            problem_text: Problem description
            
        Returns:
            Session ID
        """
        session_id = f"{problem_id}_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Create problem log file
        log_file = os.path.join(self.log_dir, "sessions", f"{session_id}.log")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"REACT AGENT SESSION\n")
            f.write("="*80 + "\n")
            f.write(f"Problem ID: {problem_id}\n")
            f.write(f"Start Time: {self.session_start.isoformat()}\n")
            f.write(f"Problem: {problem_text}\n")
            f.write("="*80 + "\n\n")
        
        return session_id
    
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
        log_file = os.path.join(self.log_dir, "sessions", f"{session_id}.log")
        
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
                    self.log_dir, "dsl_code", 
                    f"{session_id}_iter{iteration}.txt"
                )
                with open(dsl_file, 'w', encoding='utf-8') as df:
                    df.write(dsl_code)
                
                f.write(f"EXECUTION RESULT:\n")
                f.write(f"{'-'*80}\n")
                if success:
                    f.write(f"✓ SUCCESS\n")
                    if image_path:
                        f.write(f"Image saved: {image_path}\n")
                        # Copy image to log directory
                        if self.save_images and os.path.exists(image_path):
                            dest_path = os.path.join(
                                self.log_dir, "images",
                                f"{session_id}_iter{iteration}.png"
                            )
                            shutil.copy2(image_path, dest_path)
                            f.write(f"Image copied to: {dest_path}\n")
                else:
                    f.write(f"✗ FAILED\n")
                    if error:
                        # Format error message for better readability
                        f.write(f"Error Details:\n")
                        if "Error at line" in error:
                            # Enhanced error with line information
                            f.write(f"{error}\n")
                        else:
                            # Regular error
                            f.write(f"{error}\n")
                print(validation_result)
                # Log validation results if available
                if validation_result:
                    f.write(f"\nVALIDATION RESULTS:\n")
                    f.write(f"{'-'*80}\n")
                    
                    # Check for validation error first
                    if validation_result.get('error_message'):
                        f.write(f"✗ VALIDATION ERROR:\n")
                        f.write(f"{validation_result['error_message']}\n")
                        f.write(f"\n")
                    else:
                        # Normal validation results
                        f.write(f"Object Score: {validation_result.get('object_score', 0):.1%}\n")
                        f.write(f"Condition Score: {validation_result.get('condition_score', 0):.1%}\n")
                        f.write(f"Total Score: {validation_result.get('total_score', 0):.1%}\n")
                        f.write(f"Success: {'✓ YES' if validation_result.get('success', False) else '✗ NO'}\n")
                        
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
                                        f.write(f"  • {obj_type}: {objs}\n")
                        
                        # Missing objects
                        missing_objects = validation_result.get('missing_objects', {})
                        has_missing = any(objs for objs in missing_objects.values() if objs)
                        if has_missing:
                            f.write(f"\n📋 Missing Objects:\n")
                            for obj_type, objs in missing_objects.items():
                                if objs:
                                    f.write(f"  • {obj_type}: {objs}\n")
                        
                        # Passed and Failed conditions with detailed messages
                        cond_details = validation_result.get('details', {}).get('condition_details', [])
                        failed_conditions = validation_result.get('failed_conditions', [])
                        
                        # Show passed conditions
                        passed_details = [d for d in cond_details if d.get('passed', False)]
                        if passed_details:
                            f.write(f"\n✅ Passed Conditions ({len(passed_details)} total):\n")
                            for detail in passed_details:
                                cond = detail.get('condition', {})
                                message = detail.get('message', 'No message')
                                cond_type = cond.get('type', 'unknown')
                                f.write(f"  ✓ {cond_type}: {message}\n")
                        
                        # Show failed conditions with enhanced details
                        failed_details = [d for d in cond_details if not d.get('passed', False)]
                        if failed_details:
                            f.write(f"\n❌ Failed Conditions ({len(failed_details)} total):\n")
                            for detail in failed_details:
                                cond = detail.get('condition', {})
                                message = detail.get('message', 'No message')
                                cond_type = cond.get('type', 'unknown')
                                f.write(f"  ✗ {cond_type}: {message}\n")
                                
                                # Show additional parameters for context
                                for key, value in cond.items():
                                    if key != 'type' and value:
                                        f.write(f"      {key}: {value}\n")
                        
                        # Show success message if all passed (and no error occurred)
                        if not has_missing and not failed_conditions and validation_result.get('total_score', 0) > 0:
                            f.write(f"\n✅ All objects and conditions satisfied!\n")
                        elif validation_result.get('total_score', 0) == 0 and not has_missing and not failed_conditions:
                            f.write(f"\n⚠️  Low scores but no detailed error information available.\n")
                            f.write(f"    This may indicate a validation error occurred.\n")
            else:
                f.write(f"{dsl_code}\n")
            
            f.write(f"\n")
    
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
        log_file = os.path.join(self.log_dir, "sessions", f"{session_id}.log")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL RESULT\n")
            f.write(f"{'='*80}\n")
            f.write(f"End Time: {datetime.now().isoformat()}\n")
            f.write(f"Total Iterations: {iterations}\n")
            f.write(f"Success: {'✓ YES' if success else '✗ NO'}\n")
            
            if validation_result:
                f.write(f"\nVALIDATION RESULTS:\n")
                f.write(f"{'-'*80}\n")
                
                # Check for validation error first
                if validation_result.get('error_message'):
                    f.write(f"✗ VALIDATION ERROR:\n")
                    f.write(f"{validation_result['error_message']}\n")
                    f.write(f"\n")
                else:
                    f.write(f"Object Score: {validation_result.get('object_score', 0):.1%}\n")
                    f.write(f"Condition Score: {validation_result.get('condition_score', 0):.1%}\n")
                    f.write(f"Total Score: {validation_result.get('total_score', 0):.1%}\n")
                    f.write(f"Success: {'✓ YES' if validation_result.get('success', False) else '✗ NO'}\n")
                    
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
                                    f.write(f"  • {obj_type}: {objs}\n")
                    
                    # Missing objects
                    missing_objects = validation_result.get('missing_objects', {})
                    has_missing = any(objs for objs in missing_objects.values() if objs)
                    if has_missing:
                        f.write(f"\n📋 Missing Objects:\n")
                        for obj_type, objs in missing_objects.items():
                            if objs:
                                f.write(f"  • {obj_type}: {objs}\n")
                    
                    # Passed and Failed conditions with detailed messages
                    cond_details = validation_result.get('details', {}).get('condition_details', [])
                    
                    # Show passed conditions
                    passed_details = [d for d in cond_details if d.get('passed', False)]
                    if passed_details:
                        f.write(f"\n✅ Passed Conditions ({len(passed_details)} total):\n")
                        for detail in passed_details:
                            cond = detail.get('condition', {})
                            message = detail.get('message', 'No message')
                            cond_type = cond.get('type', 'unknown')
                            f.write(f"  ✓ {cond_type}: {message}\n")
                    
                    # Show failed conditions with enhanced details
                    failed_details = [d for d in cond_details if not d.get('passed', False)]
                    if failed_details:
                        f.write(f"\n❌ Failed Conditions ({len(failed_details)} total):\n")
                        for detail in failed_details:
                            cond = detail.get('condition', {})
                            message = detail.get('message', 'No message')
                            cond_type = cond.get('type', 'unknown')
                            f.write(f"  ✗ {cond_type}: {message}\n")
                            
                            # Show additional parameters for context
                            for key, value in cond.items():
                                if key != 'type' and value:
                                    f.write(f"      {key}: {value}\n")
                    
            
            f.write(f"\n{'='*80}\n")
    
    def create_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Create a summary JSON for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dictionary
        """
        log_file = os.path.join(self.log_dir, "sessions", f"{session_id}.log")
        
        if not os.path.exists(log_file):
            return {}
        
        # Count iterations and images
        dsl_files = [
            f for f in os.listdir(os.path.join(self.log_dir, "dsl_code"))
            if f.startswith(session_id)
        ]
        
        image_files = []
        if self.save_images:
            image_dir = os.path.join(self.log_dir, "images")
            if os.path.exists(image_dir):
                image_files = [
                    f for f in os.listdir(image_dir)
                    if f.startswith(session_id)
                ]
        
        summary = {
            "session_id": session_id,
            "log_file": log_file,
            "iterations": len(dsl_files),
            "dsl_files": [os.path.join(self.log_dir, "dsl_code", f) for f in sorted(dsl_files)],
            "images": [os.path.join(self.log_dir, "images", f) for f in sorted(image_files)],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary JSON
        summary_file = os.path.join(self.log_dir, "sessions", f"{session_id}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_all_sessions(self) -> list:
        """Get list of all logged sessions."""
        sessions_dir = os.path.join(self.log_dir, "sessions")
        if not os.path.exists(sessions_dir):
            return []
        
        log_files = [f for f in os.listdir(sessions_dir) if f.endswith('.log')]
        return sorted([f.replace('.log', '') for f in log_files])


# Test function
if __name__ == "__main__":
    print("="*70)
    print("Agent Logger Test")
    print("="*70)
    print()
    
    # Create logger
    logger = AgentLogger(log_dir="test_agent_logs")
    
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
    print(f"\nSummary:")
    print(json.dumps(summary, indent=2))
    
    print(f"\n✓ Logger test complete!")
    print(f"Check logs in: test_agent_logs/sessions/{session_id}.log")

