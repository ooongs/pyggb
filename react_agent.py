#!/usr/bin/env python3
"""
ReAct Agent for Geometry Problem Solving
Uses ReAct (Reasoning and Acting) pattern with multimodal observations.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
from benchmark_dataset import BenchmarkProblem
from dsl_executor import DSLExecutor, DSLExecutionResult
from multimodal_interface import MultimodalInterface, MultimodalMessage
from agent_memory import AgentMemory, Thought, Action, Observation
from dsl_validator import DSLValidator
from agent_logger import AgentLogger


class ReActAgent:
    """ReAct agent for geometry problem solving."""
    
    def __init__(self, model: str = "gpt-4o", max_iterations: int = 10,
                 save_images: bool = True, image_dir: str = "agent_images",
                 log_dir: str = "agent_logs",
                 verbose: bool = False):
        """
        Initialize ReAct agent.
        
        Args:
            model: LLM model to use
            max_iterations: Maximum ReAct iterations
            save_images: Whether to save images
            image_dir: Directory for images
            log_dir: Directory for logs
            verbose: Print detailed logs
        """
        self.model = model
        self.max_iterations = max_iterations
        self.save_images = save_images
        self.verbose = verbose
        
        # Initialize components
        self.multimodal = MultimodalInterface(model=model)
        self.executor = DSLExecutor(save_images=save_images, image_dir=image_dir)
        self.validator = DSLValidator()
        self.logger = AgentLogger(log_dir=log_dir, save_images=save_images)
        
        # Load prompts
        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.react_template = self._load_prompt("react_template.txt")
        self.dsl_guidelines = self._load_prompt("dsl_guidelines.txt")
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        prompt_path = os.path.join("prompts", filename)
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def solve(self, problem: BenchmarkProblem) -> Dict[str, Any]:
        """
        Solve a geometry problem using ReAct loop.
        
        Args:
            problem: BenchmarkProblem to solve
            
        Returns:
            Dictionary with results
        """
        # Initialize memory
        memory = AgentMemory(problem.id, problem.subject)
        
        # Start logging
        session_id = self.logger.start_problem(problem.id, problem.subject)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Solving Problem: {problem.id}")
            print(f"{'='*70}")
            print(f"Problem: {problem.subject[:100]}...")
            print(f"Session ID: {session_id}")
            print()
        
        # Main ReAct loop
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Generate thought and action
            thought_text, action_type, action_content = self._react_step(
                problem, memory, iteration
            )
            
            if self.verbose:
                print(f"Thought: {thought_text[:100]}...")
                print(f"Action: {action_type}")
            
            # Create thought and action objects
            thought = Thought(content=thought_text)
            action = Action(action_type=action_type, content=action_content)
            
            # Check if final answer
            if action_type == "final_answer":
                # Validate final DSL
                last_dsl = memory.final_dsl or action_content
                validation = self._validate_solution(last_dsl, problem)
                
                observation = Observation(
                    success=validation["success"],
                    has_image=False,
                    error=None,
                    validation_result=validation
                )
                
                memory.add_step(thought, action, observation)
                memory.final_success = validation["success"]
                memory.final_dsl = last_dsl
                
                if self.verbose:
                    print(f"Final validation: {validation['success']}")
                    print(f"Score: {validation['total_score']:.2%}")
                
                break
            
            # Execute DSL
            if action_type in ["generate_dsl", "modify_dsl"]:
                if self.verbose:
                    print(f"\nDSL Code:")
                    print("```")
                    print(action_content[:500] + ("..." if len(action_content) > 500 else ""))
                    print("```")
                
                exec_result = self.executor.execute(
                    action_content, 
                    problem_id=problem.id,
                    iteration=iteration
                )
                
                # Validate the DSL if execution was successful
                validation_result = None
                if exec_result.success:
                    validation_result = self._validate_solution(action_content, problem)
                    
                    if self.verbose:
                        print(f"\nValidation Scores:")
                        print(f"  Object Score: {validation_result['object_score']:.1%}")
                        print(f"  Condition Score: {validation_result['condition_score']:.1%}")
                        print(f"  Total Score: {validation_result['total_score']:.1%}")
                        
                        if validation_result.get('failed_conditions'):
                            print(f"  Failed Conditions: {len(validation_result['failed_conditions'])}")
                
                # Create observation with validation result
                observation = Observation(
                    success=exec_result.success,
                    has_image=exec_result.image_base64 is not None,
                    error=exec_result.error,
                    image_base64=exec_result.image_base64,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    validation_result=validation_result
                )
                
                # Store successful DSL
                if exec_result.success:
                    memory.final_dsl = action_content
                    
                    # Check if validation passes - can stop early
                    if validation_result and validation_result['success']:
                        memory.final_success = True
                        if self.verbose:
                            print(f"\n✓ Validation passed! Construction is complete.")
                
                memory.add_step(thought, action, observation)
                
                # Log this iteration
                self.logger.log_iteration(
                    session_id,
                    iteration=iteration,
                    thought=thought_text,
                    action_type=action_type,
                    dsl_code=action_content,
                    success=exec_result.success,
                    error=exec_result.error,
                    image_path=exec_result.image_path,
                    validation_result=validation_result
                )
                
                if self.verbose:
                    print(f"\nExecution: {'✓ Success' if exec_result.success else '✗ Failed'}")
                    if exec_result.error:
                        print(f"Error details:")
                        print(exec_result.error[:500] + ("..." if len(exec_result.error) > 500 else ""))
                
                # Early stop if validation passes
                if validation_result and validation_result['success']:
                    if self.verbose:
                        print(f"\n{'='*70}")
                        print(f"Problem solved successfully in {iteration} iterations!")
                        print(f"{'='*70}")
                    break
        
        # Validate final solution if we have DSL
        validation_result = None
        if memory.final_dsl:
            validation_result = self._validate_solution(memory.final_dsl, problem)
        
        # Log final result
        self.logger.log_final_result(
            session_id,
            success=memory.final_success,
            iterations=len(memory.steps),
            validation_result=validation_result
        )
        
        # Create session summary
        log_summary = self.logger.create_session_summary(session_id)
        
        # Get final results
        results = {
            "problem_id": problem.id,
            "success": memory.final_success,
            "iterations": len(memory.steps),
            "final_dsl": memory.final_dsl,
            "summary": memory.get_summary(),
            "session_id": session_id,
            "log_file": log_summary.get("log_file"),
            "images": log_summary.get("images", [])
        }
        
        # Save memory
        if self.save_images:
            memory_path = os.path.join(
                self.executor.image_dir,
                f"{problem.id}_memory.json"
            )
            memory.save_to_file(memory_path)
            results["memory_path"] = memory_path
        
        return results
    
    def _react_step(self, problem: BenchmarkProblem, memory: AgentMemory,
                   iteration: int) -> Tuple[str, str, str]:
        """
        Execute one ReAct step.
        
        Returns:
            Tuple of (thought_text, action_type, action_content)
        """
        # Build prompt
        if iteration == 1:
            # First iteration - no history
            prompt = self.react_template.format(
                problem_text=problem.subject,
                history="This is your first attempt."
            )
        else:
            # Include recent history with analysis
            history_text = self._format_history(memory, max_steps=3)
            
            # Add progress summary and failure analysis
            progress_summary = memory.get_progress_summary()
            
            full_history = f"{progress_summary}\n\n{'='*70}\n\n**Detailed History:**\n\n{history_text}"
            
            prompt = self.react_template.format(
                problem_text=problem.subject,
                history=full_history
            )
        
        # Get recent images (last 2-3 iterations)
        recent_images = []
        max_images = 3
        for step in reversed(memory.steps[-max_images:]):
            if step.observation.has_image and step.observation.image_base64:
                recent_images.append({
                    'iteration': step.iteration,
                    'success': step.observation.success,
                    'image': step.observation.image_base64
                })
        recent_images.reverse()  # 시간순으로 정렬
        
        # Create multimodal message with context
        message_text = prompt
        if recent_images:
            message_text += "\n\n**Recent Rendered Images (for comparison):**\n"
            for img_info in recent_images:
                status = "✓ Success" if img_info['success'] else "✗ Failed"
                message_text += f"- Iteration {img_info['iteration']}: {status}\n"
        
        message = MultimodalMessage(text=message_text)
        
        # Add all recent images
        for img_info in recent_images:
            message.add_image(img_info['image'])
        
        # Get response from LLM
        response = self.multimodal.send_message(
            message,
            system_prompt=self.system_prompt,
            temperature=0,
            max_tokens=3000
        )
        
        # Parse response
        thought, action_type, action_content = self._parse_response(response)
        
        return thought, action_type, action_content
    
    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse LLM response to extract thought, action type, and content.
        
        Returns:
            Tuple of (thought, action_type, action_content)
        """
        # Extract thought
        thought_match = re.search(r'\*\*Thought:\*\*\s*(.*?)(?=\*\*Action:\*\*|$)', 
                                 response, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        
        # Extract action type
        action_match = re.search(r'\*\*Action:\*\*\s*(\w+)', response, re.IGNORECASE)
        action_type = action_match.group(1).lower() if action_match else "generate_dsl"
        
        # Extract DSL code from code block
        code_match = re.search(r'```(?:dsl)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            action_content = code_match.group(1).strip()
        else:
            # Try to find any code-like content
            lines = response.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if '-> ' in line or 'const ' in line:
                    code_lines.append(line.strip())
                    in_code = True
                elif in_code and line.strip():
                    code_lines.append(line.strip())
                elif in_code and not line.strip():
                    break
            
            action_content = '\n'.join(code_lines) if code_lines else response
        
        return thought, action_type, action_content
    
    def _format_history(self, memory: AgentMemory, max_steps: int = 3) -> str:
        """Format recent history for prompt with full details."""
        recent_steps = memory.steps[-max_steps:]
        
        history_parts = []
        for step in recent_steps:
            # Header with iteration and result
            result_icon = '✓' if step.observation.success else '✗'
            history_parts.append(
                f"**Iteration {step.iteration}: {result_icon} {'Success' if step.observation.success else 'Failed'}**\n"
            )
            
            # Full thought content
            history_parts.append(f"**Thought:**\n{step.thought.content}\n")
            
            # Action type
            history_parts.append(f"**Action:** {step.action.action_type}")
            
            # Include full DSL code if available
            if step.action.action_type in ['generate_dsl', 'modify_dsl']:
                history_parts.append(f"\n**DSL Code:**\n```\n{step.action.content}\n```\n")
            
            # Observation details
            history_parts.append(f"**Observation:**")
            
            if step.observation.success:
                history_parts.append("- Execution successful")
                if step.observation.has_image:
                    history_parts.append("- Image rendered successfully")
            else:
                history_parts.append("- Execution failed")
            
            # Full error message if present
            if step.observation.error:
                history_parts.append(f"\n**Error Details:**\n```\n{step.observation.error}\n```")
            
            # Validation results if present
            if step.observation.validation_result:
                vr = step.observation.validation_result
                history_parts.append(f"\n**Validation Score:** {vr.get('total_score', 0):.1%}")
                history_parts.append(f"  - Object Score: {vr.get('object_score', 0):.1%}")
                history_parts.append(f"  - Condition Score: {vr.get('condition_score', 0):.1%}")
                
                if vr.get('missing_objects'):
                    history_parts.append("\n**Missing Objects:**")
                    for obj_type, objs in vr['missing_objects'].items():
                        if objs:
                            history_parts.append(f"  - {obj_type}: {objs}")
                            # Add helpful suggestions for fixing
                            if obj_type == "points":
                                history_parts.append(f"    💡 Fix: Add missing points - e.g., `point :  -> {objs[0]}`")
                            elif obj_type == "segments":
                                for seg in objs[:2]:  # Show first 2
                                    history_parts.append(f"    💡 Fix: Add segment - `segment : {seg[0]} {seg[1]} -> seg_{seg[0]}{seg[1]}`")
                            elif obj_type == "lines":
                                for line in objs[:2]:  # Show first 2
                                    history_parts.append(f"    💡 Fix: Add line - `line : {line[0]} {line[1]} -> line_{line[0]}{line[1]}`")
                            elif obj_type == "polygons":
                                for poly in objs[:1]:  # Show first 1
                                    pts = ' '.join(poly)
                                    labels = f"poly {' '.join(['side_' + str(i) for i in range(len(poly))])}"
                                    history_parts.append(f"    💡 Fix: Add polygon - `polygon : {pts} -> {labels}`")
                
                if vr.get('failed_conditions'):
                    history_parts.append(f"\n**Failed Conditions:** {len(vr['failed_conditions'])} conditions not met")
                    for fc in vr['failed_conditions']:
                        cond_type = fc.get('type', 'unknown')
                        history_parts.append(f"  - Type: {cond_type}")
                        
                        # Show validation message if available
                        validation_msg = fc.get('validation_message')
                        if validation_msg:
                            history_parts.append(f"    Result: {validation_msg}")
                        
                        # Show specific details and helpful fixes based on condition type
                        if cond_type == 'angle_value':
                            points = fc.get('points', [])
                            expected_value = fc.get('value', 'N/A')
                            history_parts.append(f"    Points: {points}")
                            history_parts.append(f"    Expected angle: {expected_value}°")
                            if not validation_msg:
                                history_parts.append(f"    Issue: The angle does not match the required value")
                            if points and len(points) > 0 and len(points[0]) == 3:
                                p1, vertex, p3 = points[0]
                                history_parts.append(f"    💡 Fix: Use rotation to create angle - `rotate : {p1} {expected_value}° {vertex} -> {p3}`")
                        
                        elif cond_type == 'parallel':
                            objects = fc.get('objects', [])
                            history_parts.append(f"    Objects: {objects}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: These lines should be parallel but are not")
                            if len(objects) == 2:
                                line1, line2 = objects
                                history_parts.append(f"    💡 Fix: Construct {line2[0]}{line2[1]} parallel to {line1[0]}{line1[1]} using same direction vector")
                        
                        elif cond_type == 'perpendicular':
                            objects = fc.get('objects', [])
                            history_parts.append(f"    Objects: {objects}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: These lines should be perpendicular but are not")
                            if len(objects) == 2:
                                line1, line2 = objects
                                history_parts.append(f"    💡 Fix: Use `orthogonal_line : {line2[0]} line_{line1[0]}{line1[1]} -> perp_line`")
                        
                        elif cond_type == 'point_on_segment':
                            point = fc.get('point', 'P')
                            segment = fc.get('segment', [])
                            history_parts.append(f"    Point: {point}, Segment: {segment}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: Point {point} should be on segment {segment[0]}{segment[1]}")
                            history_parts.append(f"    💡 Fix: Place {point} between {segment[0]} and {segment[1]} (use coordinates or intersection)")
                        
                        elif cond_type == 'segment_equality':
                            segments = fc.get('segments', [])
                            history_parts.append(f"    Segments: {segments}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: These segments should have equal length but do not")
                            history_parts.append(f"    💡 Fix: Use same distance for both segments or construct them symmetrically")
                        
                        elif cond_type == 'angle_bisector':
                            line = fc.get('line', [])
                            angle_points = fc.get('angle_points', [])
                            history_parts.append(f"    Bisector line: {line}")
                            history_parts.append(f"    Angle: {angle_points}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: The line should bisect the angle but does not")
                            if len(angle_points) == 3:
                                history_parts.append(f"    💡 Fix: Use `angular_bisector : {angle_points[0]} {angle_points[1]} {angle_points[2]} -> bisector`")
                        
                        elif cond_type == 'collinear':
                            points = fc.get('points', [])
                            history_parts.append(f"    Points: {points}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: These points should be collinear but are not")
                            history_parts.append(f"    💡 Fix: Place all points on the same line")
                        
                        elif cond_type == 'not_collinear':
                            points = fc.get('points', [])
                            history_parts.append(f"    Points: {points}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: These points should NOT be collinear (degenerate triangle)")
                            history_parts.append(f"    💡 Fix: Ensure points form a valid triangle, not a line")
                        
                        elif cond_type == 'midpoint_of':
                            point = fc.get('point', 'M')
                            segment = fc.get('segment', [])
                            history_parts.append(f"    Point: {point}, Segment: {segment}")
                            if not validation_msg:
                                history_parts.append(f"    Issue: Point {point} should be midpoint of {segment[0]}{segment[1]}")
                            history_parts.append(f"    💡 Fix: Use `midpoint : {segment[0]} {segment[1]} -> {point}`")
                        
                        else:
                            # Generic condition display
                            if not validation_msg:
                                history_parts.append(f"    Details: {fc}")
                            history_parts.append(f"    💡 Review the condition and adjust your construction accordingly")
            
            history_parts.append("\n" + "="*70 + "\n")
        
        return "\n".join(history_parts)
    
    def _validate_solution(self, dsl_code: str, problem: BenchmarkProblem) -> Dict[str, Any]:
        """Validate solution against problem requirements."""
        # Create temporary DSL file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(dsl_code)
            dsl_file = f.name
        
        try:
            validation = self.validator.validate(dsl_file, problem)
            
            # Extract detailed messages from validation details
            failed_conditions_with_messages = []
            if validation.details and 'condition_details' in validation.details:
                for detail in validation.details['condition_details']:
                    if not detail.get('passed', False):
                        condition = detail.get('condition', {})
                        message = detail.get('message', 'No message')
                        
                        # Combine condition info with the validation message
                        failed_cond = dict(condition)
                        failed_cond['validation_message'] = message
                        failed_conditions_with_messages.append(failed_cond)
            else:
                # Fallback to basic failed conditions
                failed_conditions_with_messages = [c for c in validation.failed_conditions]
            
            return {
                "success": validation.success,
                "total_score": validation.total_score,
                "object_score": validation.object_score,
                "condition_score": validation.condition_score,
                "missing_objects": validation.missing_objects,
                "failed_conditions": failed_conditions_with_messages
            }
        finally:
            if os.path.exists(dsl_file):
                os.remove(dsl_file)


# Test function
if __name__ == "__main__":
    from benchmark_dataset import BenchmarkDataset, RequiredObjects, ConditionBuilder, VerificationCondition
    
    print("="*70)
    print("ReAct Agent Test")
    print("="*70)
    print()
    
    # Create a simple test problem
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]]
    )
    
    conditions = [
        VerificationCondition.from_dict(
            ConditionBuilder.not_collinear(["A", "B", "C"])
        )
    ]
    
    from benchmark_dataset import BenchmarkProblem
    problem = BenchmarkProblem(
        id="test_triangle",
        subject="Create a triangle with vertices A, B, and C",
        required_objects=required_objects,
        verification_conditions=conditions
    )
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Set it in .env file to run the agent.")
        print("For now, skipping agent test.")
    else:
        print("Testing ReAct Agent...")
        agent = ReActAgent(model="gpt-4o", max_iterations=3, verbose=True)
        
        results = agent.solve(problem)
        
        print("\n" + "="*70)
        print("Results:")
        print(f"  Success: {results['success']}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Memory saved: {results.get('memory_path')}")

