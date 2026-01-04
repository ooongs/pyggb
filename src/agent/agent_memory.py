#!/usr/bin/env python3
"""
Agent Memory for ReAct Agent
Manages conversation history, previous attempts, and learning from failures.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np


def _convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-serializable types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None

    # Handle numpy types
    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj) if isinstance(obj, np.bool_) else int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle Python bool explicitly
    elif isinstance(obj, bool):
        return bool(obj)

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]

    # Return as-is for primitives and other types
    else:
        return obj


@dataclass
class Thought:
    """A single thought in the ReAct loop."""
    content: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Action:
    """An action taken by the agent."""
    action_type: str  # generate_dsl, modify_dsl, validate, final_answer
    content: str  # DSL code or other content
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Observation:
    """Observation from environment after action."""
    success: bool
    has_image: bool
    error: Optional[str]
    validation_result: Optional[Dict] = None
    image_base64: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_text(self) -> str:
        """Convert observation to text for LLM."""
        parts = []
        
        if self.success:
            parts.append("✓ DSL executed successfully")
            if self.has_image:
                parts.append("Image rendered successfully")
        else:
            parts.append("✗ DSL execution failed")
            if self.error:
                parts.append(f"Error: {self.error}")
        
        if self.validation_result:
            vr = self.validation_result
            parts.append(f"\nValidation: {vr.get('total_score', 0):.1%} score")
            if vr.get('missing_objects'):
                missing = vr['missing_objects']
                for obj_type, objs in missing.items():
                    if objs:
                        parts.append(f"  Missing {obj_type}: {objs}")
            if vr.get('failed_conditions'):
                parts.append(f"  Failed conditions: {len(vr['failed_conditions'])}")
        
        if self.stdout and self.stdout.strip():
            parts.append(f"\nOutput: {self.stdout[:200]}")
        
        return "\n".join(parts)


@dataclass
class Step:
    """A single step in the ReAct loop."""
    iteration: int
    thought: Thought
    action: Action
    observation: Observation
    
    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "thought": asdict(self.thought),
            "action": asdict(self.action),
            "observation": {
                **asdict(self.observation),
                "image_base64": None  # Exclude large base64 from dict
            }
        }


class AgentMemory:
    """Memory system for ReAct agent."""
    
    def __init__(self, problem_id: str, problem_text: str):
        """
        Initialize agent memory for a problem.
        
        Args:
            problem_id: Problem identifier
            problem_text: Problem description
        """
        self.problem_id = problem_id
        self.problem_text = problem_text
        self.steps: List[Step] = []
        self.start_time = datetime.now()
        self.end_time = None
        self.final_success = False
        self.final_dsl = None
    
    def add_step(self, thought: Thought, action: Action, observation: Observation):
        """Add a ReAct step to memory."""
        step = Step(
            iteration=len(self.steps) + 1,
            thought=thought,
            action=action,
            observation=observation
        )
        self.steps.append(step)
    
    def get_conversation_history(self, include_images: bool = True,
                                 max_steps: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history formatted for LLM.
        
        Args:
            include_images: Whether to include images in history
            max_steps: Maximum number of recent steps to include
            
        Returns:
            List of conversation messages
        """
        steps_to_include = self.steps
        if max_steps:
            steps_to_include = self.steps[-max_steps:]
        
        history = []
        
        for step in steps_to_include:
            # Add thought + action as user message
            user_content = []
            
            # Text part
            text = f"**Thought {step.iteration}:** {step.thought.content}\n\n"
            text += f"**Action:** {step.action.action_type}\n"
            
            if step.action.action_type in ['generate_dsl', 'modify_dsl']:
                text += f"```\n{step.action.content}\n```"
            else:
                text += step.action.content
            
            user_content.append({"type": "text", "text": text})
            
            history.append({"role": "user", "content": user_content})
            
            # Add observation as assistant message
            assistant_content = []
            
            obs_text = f"**Observation {step.iteration}:**\n{step.observation.to_text()}"
            assistant_content.append({"type": "text", "text": obs_text})
            
            # Add image if available and requested
            if include_images and step.observation.has_image and step.observation.image_base64:
                assistant_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{step.observation.image_base64}"
                    }
                })
            
            history.append({"role": "assistant", "content": assistant_content})
        
        return history
    
    def get_recent_failures(self, n: int = 3) -> List[str]:
        """Get recent failure messages for learning."""
        failures = []
        for step in reversed(self.steps):
            if not step.observation.success and step.observation.error:
                failures.append(step.observation.error)
                if len(failures) >= n:
                    break
        return failures
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure patterns to help agent learn."""
        analysis = {
            "total_failures": 0,
            "common_errors": {},
            "failure_rate": 0.0,
            "repeated_mistakes": []
        }
        
        error_types = {}
        failed_dsl_snippets = []
        
        for step in self.steps:
            if not step.observation.success:
                analysis["total_failures"] += 1
                
                # Categorize error types
                if step.observation.error:
                    error = step.observation.error
                    
                    # Extract error type (first line usually has the error type)
                    error_lines = error.split('\n')
                    error_type = error_lines[0] if error_lines else "Unknown"
                    
                    # Count error types
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    # Store failed DSL for pattern detection
                    if step.action.action_type in ['generate_dsl', 'modify_dsl']:
                        failed_dsl_snippets.append({
                            'iteration': step.iteration,
                            'error': error_type,
                            'dsl': step.action.content[:200]  # First 200 chars
                        })
        
        # Find most common errors
        if error_types:
            analysis["common_errors"] = dict(sorted(
                error_types.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5 errors
        
        # Calculate failure rate
        if self.steps:
            analysis["failure_rate"] = analysis["total_failures"] / len(self.steps)
        
        # Detect repeated mistakes (same error type appearing multiple times)
        for error_type, count in error_types.items():
            if count >= 2:
                analysis["repeated_mistakes"].append({
                    "error": error_type,
                    "count": count
                })
        
        return analysis
    
    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary for the agent."""
        if not self.steps:
            return "No attempts yet."
        
        success_count = sum(1 for step in self.steps if step.observation.success)
        failure_count = len(self.steps) - success_count
        
        summary_parts = [
            f"**Progress Summary (Total: {len(self.steps)} iterations)**",
            f"- Successful executions: {success_count}",
            f"- Failed executions: {failure_count}"
        ]
        
        # Add recent trend
        if len(self.steps) >= 3:
            recent_results = [step.observation.success for step in self.steps[-3:]]
            if all(recent_results):
                summary_parts.append("- Recent trend: ✓ All recent attempts successful!")
            elif not any(recent_results):
                summary_parts.append("- Recent trend: ✗ All recent attempts failed - try a different approach")
            else:
                summary_parts.append("- Recent trend: ⚡ Mixed results - getting closer")
        
        # Add failure analysis if there are failures
        if failure_count > 0:
            analysis = self.get_failure_analysis()
            if analysis["common_errors"]:
                summary_parts.append("\n**Most Common Errors:**")
                for error, count in list(analysis["common_errors"].items())[:3]:
                    summary_parts.append(f"  - {error[:80]} (occurred {count}x)")
            
            if analysis["repeated_mistakes"]:
                summary_parts.append("\n**⚠️ Repeated Mistakes - Avoid These:**")
                for mistake in analysis["repeated_mistakes"][:2]:
                    summary_parts.append(f"  - {mistake['error'][:80]} (failed {mistake['count']}x)")
        
        return "\n".join(summary_parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of episode."""
        if self.end_time is None:
            self.end_time = datetime.now()
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        success_count = sum(1 for step in self.steps if step.observation.success)
        
        return {
            "problem_id": self.problem_id,
            "problem_text": self.problem_text[:100] + "..." if len(self.problem_text) > 100 else self.problem_text,
            "total_steps": len(self.steps),
            "successful_executions": success_count,
            "duration_seconds": duration,
            "final_success": self.final_success,
            "has_final_dsl": self.final_dsl is not None
        }
    
    def save_to_file(self, filepath: str):
        """Save memory to JSON file."""
        data = {
            "problem_id": self.problem_id,
            "problem_text": self.problem_text,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "final_success": self.final_success,
            "final_dsl": self.final_dsl,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.get_summary()
        }

        # Convert to JSON-serializable format (handles numpy types)
        data = _convert_to_json_serializable(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str, max_iteration: Optional[int] = None) -> 'AgentMemory':
        """
        Load memory from JSON file with optional iteration truncation.

        Args:
            filepath: Path to memory JSON file
            max_iteration: Load steps up to this iteration (inclusive). None = load all

        Returns:
            AgentMemory with reconstructed steps
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create memory instance
        memory = cls(data['problem_id'], data['problem_text'])
        memory.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            memory.end_time = datetime.fromisoformat(data['end_time'])
        memory.final_success = data.get('final_success', False)
        memory.final_dsl = data.get('final_dsl')

        # Reconstruct steps from JSON
        for step_data in data.get('steps', []):
            iteration = step_data['iteration']

            # Stop if max_iteration is specified and we've reached it
            if max_iteration is not None and iteration > max_iteration:
                break

            # Reconstruct Thought
            thought_data = step_data['thought']
            thought = Thought(
                content=thought_data['content'],
                timestamp=thought_data.get('timestamp')
            )

            # Reconstruct Action
            action_data = step_data['action']
            action = Action(
                action_type=action_data['action_type'],
                content=action_data['content'],
                timestamp=action_data.get('timestamp')
            )

            # Reconstruct Observation
            obs_data = step_data['observation']
            observation = Observation(
                success=obs_data['success'],
                has_image=obs_data['has_image'],
                error=obs_data.get('error'),
                validation_result=obs_data.get('validation_result'),
                image_base64=obs_data.get('image_base64'),  # Will be None (excluded in save)
                stdout=obs_data.get('stdout', ''),
                stderr=obs_data.get('stderr', ''),
                timestamp=obs_data.get('timestamp')
            )

            # Create Step and add to memory
            step = Step(
                iteration=iteration,
                thought=thought,
                action=action,
                observation=observation
            )
            memory.steps.append(step)

        return memory


# Test function
if __name__ == "__main__":
    print("="*70)
    print("Agent Memory Test")
    print("="*70)
    print()
    
    # Create test memory
    memory = AgentMemory(
        problem_id="test_1",
        problem_text="Create a triangle with parallel lines"
    )
    
    # Add some steps
    thought1 = Thought("I need to define points first")
    action1 = Action("generate_dsl", "point :  -> A\npoint :  -> B")
    obs1 = Observation(success=True, has_image=True, error=None)
    memory.add_step(thought1, action1, obs1)
    
    thought2 = Thought("Now I need to add a third point")
    action2 = Action("modify_dsl", "point :  -> A\npoint :  -> B\npoint :  -> C")
    obs2 = Observation(success=False, has_image=False, error="Syntax error on line 3")
    memory.add_step(thought2, action2, obs2)
    
    # Get summary
    summary = memory.get_summary()
    print("Memory Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Added {len(memory.steps)} steps to memory")
    print(f"✓ Recent failures: {len(memory.get_recent_failures())}")

