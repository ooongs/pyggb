#!/usr/bin/env python3
"""
ReAct Agent for Geometry Problem Solving
Uses ReAct (Reasoning and Acting) pattern with multimodal observations.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List
from src.benchmark.benchmark_dataset import BenchmarkProblem
from src.dsl.dsl_executor import DSLExecutor, DSLExecutionResult
from src.interfaces.multimodal_interface import MultimodalInterface, MultimodalMessage
from src.agent.agent_memory import AgentMemory, Thought, Action, Observation
from src.dsl.dsl_validator import DSLValidator
from src.agent.agent_logger import AgentLogger
from src.utils import get_prompts_dir, get_hints_dir, get_output_dir


class ErrorHintManager:
    """Manages error-specific hints for DSL execution failures."""

    # Error patterns mapped to hint file names
    ERROR_PATTERNS = {
        "undefined": [
            r"not defined",
            r"undefined",
            r"not found",
            r"unknown (object|variable|label)",
            r"does not exist",
        ],
        "output_count": [
            r"output.*mismatch",
            r"expected \d+ outputs?",
            r"wrong number of (outputs?|arguments?)",
            r"polygon.*outputs?",
        ],
        "syntax": [
            r"syntax error",
            r"missing.*:",
            r"missing.*->",
            r"invalid syntax",
            r"parse error",
        ],
        "invalid_command": [
            r"invalid command",
            r"unknown command",
            r"no such command",
            r"command.*not (found|recognized)",
        ],
        "type": [
            r"type (error|mismatch)",
            r"expected.*got",
            r"incompatible type",
            r"wrong type",
        ],
        "duplicate": [
            r"duplicate",
            r"already (defined|exists)",
            r"redefined",
            r"label.*exists",
        ],
        "constraint": [
            r"cannot assert",
            r"equality.*angle",
            r"parallel.*assert",
            r"constraint",
        ],
    }

    def __init__(self, hints_dir: Optional[str] = None):
        """Initialize with hints directory path."""
        self.hints_dir = hints_dir or str(get_hints_dir())
        self._hints_cache: Dict[str, str] = {}

    def _load_hint(self, hint_name: str) -> str:
        """Load a hint file, with caching."""
        if hint_name in self._hints_cache:
            return self._hints_cache[hint_name]

        hint_path = os.path.join(self.hints_dir, f"error_{hint_name}.txt")
        if os.path.exists(hint_path):
            with open(hint_path, "r", encoding="utf-8") as f:
                hint = f.read()
                self._hints_cache[hint_name] = hint
                return hint
        return ""

    def get_hints_for_error(self, error_message: str) -> str:
        """
        Analyze error message and return relevant hints.

        Args:
            error_message: The DSL execution error message

        Returns:
            Concatenated relevant hints, or empty string if no matches
        """
        if not error_message:
            return ""

        error_lower = error_message.lower()
        matched_hints = []

        for hint_name, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    hint = self._load_hint(hint_name)
                    if hint and hint not in matched_hints:
                        matched_hints.append(hint)
                    break  # Only add each hint once

        if matched_hints:
            return (
                "\n\n---\n\n## 💡 Helpful Hints for Your Error:\n\n"
                + "\n\n---\n\n".join(matched_hints)
            )

        return ""

    def get_all_hint_names(self) -> List[str]:
        """Return list of available hint names."""
        return list(self.ERROR_PATTERNS.keys())


class ReActAgent:
    """ReAct agent for geometry problem solving."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 10,
        save_images: bool = True,
        image_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False,
        use_vision: bool = True,
        run_id: Optional[str] = None,
        resume_mode: bool = False,
        additional_prompt: Optional[str] = None,
    ):
        """
        Initialize ReAct agent.

        Args:
            model: LLM model to use
            max_iterations: Maximum ReAct iterations
            save_images: Whether to save images
            image_dir: Directory for images
            log_dir: Directory for logs
            verbose: Print detailed logs
            use_vision: Whether to send rendered images to LLM (for vision comparison)
            run_id: Custom run identifier for logging (default: timestamp)
            resume_mode: Whether this is a resume run (merge results instead of overwrite)
            additional_prompt: Additional prompt text to append to system prompt (optional)
        """
        self.model = model
        self.max_iterations = max_iterations
        self.save_images = save_images
        self.verbose = verbose
        self.use_vision = use_vision

        # Initialize components
        self.multimodal = MultimodalInterface(model=model)
        self.executor = DSLExecutor(
            save_images=save_images,
            image_dir=image_dir or str(get_output_dir("agent_images")),
        )
        self.validator = DSLValidator()
        self.logger = AgentLogger(
            log_dir=log_dir or str(get_output_dir("agent_logs")),
            save_images=save_images,
            run_id=run_id,
            verbose=verbose,
            resume_mode=resume_mode,
        )
        self.hint_manager = ErrorHintManager()

        # Store run info
        self.run_id = self.logger.run_id
        self.run_dir = self.logger.run_dir

        # Load prompts
        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.react_template = self._load_prompt("react_template.txt")

        # Append additional prompt if provided
        if additional_prompt:
            self.system_prompt += f"\n\n{additional_prompt}"

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        prompt_path = get_prompts_dir() / filename
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def solve(
        self,
        problem: BenchmarkProblem,
        resume_memory: Optional[AgentMemory] = None,
        start_iteration: int = 1,
        reexecute_last_iteration: bool = False
    ) -> Dict[str, Any]:
        """
        Solve a geometry problem using ReAct loop.

        Args:
            problem: BenchmarkProblem to solve
            resume_memory: Pre-loaded memory to resume from (optional)
            start_iteration: Iteration number to start from (default: 1)
            reexecute_last_iteration: If True, re-execute DSL from last loaded iteration (default: False)

        Returns:
            Dictionary with results
        """
        # Initialize or resume memory
        if resume_memory is not None:
            # Validate problem_id matches
            if resume_memory.problem_id != problem.id:
                raise ValueError(
                    f"Resume memory problem_id ({resume_memory.problem_id}) "
                    f"does not match current problem_id ({problem.id})"
                )
            memory = resume_memory
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"📂 RESUMING from previous session")
                print(f"{'='*70}")
                print(f"Loaded steps: {len(memory.steps)}")
                if reexecute_last_iteration and len(memory.steps) > 0:
                    print(f"Will re-execute iteration {len(memory.steps)} DSL code")
                print(f"Starting from iteration: {max(start_iteration, len(memory.steps) + 1)}")
                print(f"{'='*70}\n")
        else:
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

        # Re-execute last iteration DSL if requested
        if reexecute_last_iteration and len(memory.steps) > 0:
            last_step = memory.steps[-1]
            last_iteration = last_step.iteration

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔄 RE-EXECUTING Iteration {last_iteration} DSL")
                print(f"{'='*70}")
                print(f"Action type: {last_step.action.action_type}")
                print(f"\nDSL Code:")
                print("```")
                print(last_step.action.content[:500] + ("..." if len(last_step.action.content) > 500 else ""))
                print("```\n")

            # Get DSL code from last action
            dsl_code = last_step.action.content

            # Execute DSL
            exec_result = self.executor.execute(
                dsl_code, problem_id=problem.id, iteration=last_iteration
            )

            # Validate if successful
            validation_result = None
            if exec_result.success:
                validation_result = self._validate_solution(dsl_code, problem)
                memory.final_dsl = dsl_code

            # Load image as base64 if exists and vision is enabled
            image_base64 = None
            if self.use_vision and exec_result.image_path and os.path.exists(exec_result.image_path):
                import base64
                with open(exec_result.image_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Create new observation
            new_observation = Observation(
                success=exec_result.success,
                has_image=exec_result.image_path is not None,
                error=exec_result.error,
                validation_result=validation_result,
                image_base64=image_base64,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
            )

            # Replace last step's observation
            memory.steps[-1].observation = new_observation

            if self.verbose:
                result_str = "✓ SUCCESS" if exec_result.success else "✗ FAILED"
                print(f"Re-execution result: {result_str}")
                if exec_result.error:
                    print(f"Error: {exec_result.error}")
                if validation_result:
                    print(f"Validation score: {validation_result['total_score']:.2%}")
                print(f"{'='*70}\n")

        # Main ReAct loop
        # Calculate actual start iteration (handle resume case)
        actual_start = max(start_iteration, len(memory.steps) + 1)

        # Validate start_iteration
        if actual_start > self.max_iterations:
            raise ValueError(
                f"Start iteration ({actual_start}) exceeds max_iterations ({self.max_iterations})"
            )

        for iteration in range(actual_start, self.max_iterations + 1):
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
                    validation_result=validation,
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
                    print(
                        action_content[:500]
                        + ("..." if len(action_content) > 500 else "")
                    )
                    print("```")

                exec_result = self.executor.execute(
                    action_content, problem_id=problem.id, iteration=iteration
                )

                # Validate the DSL if execution was successful
                validation_result = None
                if exec_result.success:
                    validation_result = self._validate_solution(action_content, problem)

                    if self.verbose:
                        print(f"\nValidation Scores:")
                        print(
                            f"  Object Score: {validation_result['object_score']:.1%}"
                        )
                        print(
                            f"  Condition Score: {validation_result['condition_score']:.1%}"
                        )
                        print(f"  Total Score: {validation_result['total_score']:.1%}")

                        if validation_result.get("failed_conditions"):
                            print(
                                f"  Failed Conditions: {len(validation_result['failed_conditions'])}"
                            )

                # Create observation with validation result
                observation = Observation(
                    success=exec_result.success,
                    has_image=exec_result.image_base64 is not None,
                    error=exec_result.error,
                    image_base64=exec_result.image_base64,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    validation_result=validation_result,
                )

                # Store successful DSL
                if exec_result.success:
                    memory.final_dsl = action_content

                    # Check if validation passes - can stop early
                    if validation_result and validation_result["success"]:
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
                    validation_result=validation_result,
                )

                if self.verbose:
                    print(
                        f"\nExecution: {'✓ Success' if exec_result.success else '✗ Failed'}"
                    )
                    if exec_result.error:
                        print(f"Error details:")
                        print(
                            exec_result.error[:500]
                            + ("..." if len(exec_result.error) > 500 else "")
                        )

                # Early stop if validation passes
                if validation_result and validation_result["success"]:
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
            validation_result=validation_result,
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
            "images": log_summary.get("images", []),
            "validation_result": validation_result,  # Include validation result for error checking
        }

        # Save memory
        if self.save_images:
            memory_path = os.path.join(
                self.executor.image_dir, f"{problem.id}_memory.json"
            )
            memory.save_to_file(memory_path)
            results["memory_path"] = memory_path

        return results

    def _react_step(
        self, problem: BenchmarkProblem, memory: AgentMemory, iteration: int
    ) -> Tuple[str, str, str]:
        """
        Execute one ReAct step.

        Returns:
            Tuple of (thought_text, action_type, action_content)
        """
        # Build prompt
        if iteration == 1:
            # First iteration - no history
            prompt = self.react_template.format(
                problem_text=problem.subject, history="This is your first attempt."
            )
        else:
            # Include recent history with analysis
            history_text = self._format_history(memory, max_steps=3)

            # Add progress summary and failure analysis
            progress_summary = memory.get_progress_summary()

            full_history = f"{progress_summary}\n\n{'='*70}\n\n**Detailed History:**\n\n{history_text}"

            prompt = self.react_template.format(
                problem_text=problem.subject, history=full_history
            )

        # Get recent images (last 2-3 iterations) - only if vision is enabled
        recent_images = []
        if self.use_vision:
            max_images = 1
            for step in reversed(memory.steps[-max_images:]):
                if step.observation.has_image and step.observation.image_base64:
                    recent_images.append(
                        {
                            "iteration": step.iteration,
                            "success": step.observation.success,
                            "image": step.observation.image_base64,
                        }
                    )
            recent_images.reverse()  # 시간순으로 정렬

        # Create multimodal message with context
        message_text = prompt
        if recent_images and self.use_vision:
            message_text += "\n\n**Recent Rendered Images (for comparison):**\n"
            for img_info in recent_images:
                status = "✓ Success" if img_info["success"] else "✗ Failed"
                message_text += f"- Iteration {img_info['iteration']}: {status}\n"
        elif not self.use_vision:
            message_text += (
                "\n\n**Note: Vision is disabled. No images will be shown.**\n"
            )

        message = MultimodalMessage(text=message_text)

        # Add all recent images only if vision is enabled
        if self.use_vision:
            for img_info in recent_images:
                message.add_image(img_info["image"])

        # Get response from LLM
        response = self.multimodal.send_message(
            message, system_prompt=self.system_prompt, temperature=0, max_tokens=10000
        )

        # Normalize response to plain text (handles list/dict from OpenRouter/OpenAI)
        response_text = self._normalize_response(response)

        # Parse response
        thought, action_type, action_content = self._parse_response(response_text)

        return thought, action_type, action_content

    def _normalize_response(self, response: Any) -> str:
        """Convert LLM response into a plain string."""
        if response is None:
            raise ValueError("LLM returned no content")
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif "content" in item and isinstance(item["content"], str):
                        parts.append(item["content"])
            if parts:
                return "\n".join(parts)
            return str(response)
        if isinstance(response, dict):
            if "output_text" in response:
                return str(response["output_text"])
            if "text" in response:
                return str(response["text"])
            if "message" in response:
                return str(response["message"])
        return str(response)

    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse LLM response to extract thought, action type, and content.

        Returns:
            Tuple of (thought, action_type, action_content)
        """
        # Extract thought
        thought_match = re.search(
            r"\*\*Thought:\*\*\s*(.*?)(?=\*\*Action:\*\*|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        thought = (
            thought_match.group(1).strip() if thought_match else "No thought provided"
        )

        # Extract action type
        action_match = re.search(r"\*\*Action:\*\*\s*(\w+)", response, re.IGNORECASE)
        action_type = action_match.group(1).lower() if action_match else "generate_dsl"

        # Extract DSL code from code block
        code_match = re.search(r"```(?:dsl)?\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            action_content = code_match.group(1).strip()
        else:
            # Try to find any code-like content
            lines = response.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if "-> " in line or "const " in line:
                    code_lines.append(line.strip())
                    in_code = True
                elif in_code and line.strip():
                    code_lines.append(line.strip())
                elif in_code and not line.strip():
                    break

            action_content = "\n".join(code_lines) if code_lines else response

        return thought, action_type, action_content

    def _format_history(self, memory: AgentMemory, max_steps: int = 3) -> str:
        """Format recent history for prompt with full details."""
        recent_steps = memory.steps[-max_steps:]

        history_parts = []
        for step in recent_steps:
            # Header with iteration and result
            result_icon = "✓" if step.observation.success else "✗"
            history_parts.append(
                f"**Iteration {step.iteration}: {result_icon} {'Success' if step.observation.success else 'Failed'}**\n"
            )

            # Full thought content
            history_parts.append(f"**Thought:**\n{step.thought.content}\n")

            # Action type
            history_parts.append(f"**Action:** {step.action.action_type}")

            # Include full DSL code if available
            if step.action.action_type in ["generate_dsl", "modify_dsl"]:
                history_parts.append(
                    f"\n**DSL Code:**\n```\n{step.action.content}\n```\n"
                )

            # Observation details
            history_parts.append(f"**Observation:**")

            if step.observation.success:
                history_parts.append("- Execution successful")
                if step.observation.has_image:
                    history_parts.append("- Image rendered successfully")
            else:
                history_parts.append("- Execution failed")

            # Full error message if present, with dynamic hints
            if step.observation.error:
                history_parts.append(
                    f"\n**Error Details:**\n```\n{step.observation.error}\n```"
                )

                # Add relevant hints for the error (only for execution failures)
                error_hints = self.hint_manager.get_hints_for_error(
                    step.observation.error
                )
                if error_hints:
                    history_parts.append(error_hints)

            # Validation results if present
            if step.observation.validation_result:
                vr = step.observation.validation_result
                history_parts.append(f"\n**Validation Scores:**")
                history_parts.append(
                    f"  - Object Score: {vr.get('object_score', 0):.1%}"
                )
                history_parts.append(
                    f"  - Condition Score: {vr.get('condition_score', 0):.1%}"
                )
                history_parts.append(
                    f"  - **Total Score: {vr.get('total_score', 0):.1%}** ({'✓ PASS' if vr.get('success', False) else '✗ FAIL'})"
                )

                # Show found objects if available
                details = vr.get("details", {})
                found_objects = details.get("found_objects", {})
                if found_objects and any(
                    objs for objs in found_objects.values() if objs
                ):
                    history_parts.append(f"\n**✓ Found Objects:**")
                    for obj_type, objs in found_objects.items():
                        if objs:
                            objs_str = (
                                ", ".join(str(o) for o in objs)
                                if isinstance(objs, list)
                                else str(objs)
                            )
                            history_parts.append(
                                f"  - {obj_type.capitalize()}: {objs_str}"
                            )

                if vr.get("missing_objects"):
                    history_parts.append("\n**Missing Objects:**")
                    for obj_type, objs in vr["missing_objects"].items():
                        if objs:
                            history_parts.append(f"  - {obj_type}: {objs}")
                            # Add helpful suggestions for fixing
                            if obj_type == "points":
                                history_parts.append(
                                    f"    💡 Fix: Add missing points - e.g., `point :  -> {objs[0]}`"
                                )
                            elif obj_type == "segments":
                                for seg in objs[:2]:  # Show first 2
                                    history_parts.append(
                                        f"    💡 Fix: Add segment - `segment : {seg[0]} {seg[1]} -> seg_{seg[0]}{seg[1]}`"
                                    )
                            elif obj_type == "lines":
                                for line in objs[:2]:  # Show first 2
                                    history_parts.append(
                                        f"    💡 Fix: Add line - `line : {line[0]} {line[1]} -> line_{line[0]}{line[1]}`"
                                    )
                            elif obj_type == "polygons":
                                for poly in objs[:1]:  # Show first 1
                                    pts = " ".join(poly)
                                    labels = f"poly {' '.join(['side_' + str(i) for i in range(len(poly))])}"
                                    history_parts.append(
                                        f"    💡 Fix: Add polygon - `polygon : {pts} -> {labels}`"
                                    )

                # Show passed conditions summary
                passed_details = [
                    d
                    for d in details.get("condition_details", [])
                    if d.get("passed", False)
                ]
                if passed_details:
                    history_parts.append(
                        f"\n**✓ Passed Conditions:** {len(passed_details)} conditions satisfied"
                    )

                # Show failed conditions with detailed information
                failed_details = [
                    d
                    for d in details.get("condition_details", [])
                    if not d.get("passed", False)
                ]
                if failed_details or vr.get("failed_conditions"):
                    # Use condition_details if available, otherwise fall back to failed_conditions
                    failed_list = failed_details if failed_details else []
                    if not failed_list:
                        failed_list = [
                            {
                                "condition": fc,
                                "message": fc.get("validation_message", "No message"),
                            }
                            for fc in vr.get("failed_conditions", [])
                        ]

                    history_parts.append(
                        f"\n**✗ Failed Conditions:** {len(failed_list)} conditions NOT satisfied"
                    )

                    for idx, detail in enumerate(failed_list, 1):
                        if "condition" in detail:
                            fc = detail["condition"]
                            msg = detail.get("message", "No message")
                        else:
                            fc = detail
                            msg = fc.get("validation_message", "No message")

                        cond_type = fc.get("type", "unknown")
                        history_parts.append(f"\n  {idx}. **{cond_type}**")
                        history_parts.append(f"     Status: {msg}")

                        # Show all relevant parameters from the condition
                        params_to_show = {}
                        for key, value in fc.items():
                            if (
                                key not in ["type", "validation_message"]
                                and value is not None
                            ):
                                params_to_show[key] = value

                        if params_to_show:
                            for key, value in params_to_show.items():
                                if isinstance(value, (list, tuple)):
                                    value_str = ", ".join(str(v) for v in value)
                                elif isinstance(value, float):
                                    value_str = f"{value:.2f}"
                                else:
                                    value_str = str(value)
                                history_parts.append(f"     • {key}: {value_str}")

                        # Show helpful fix suggestions based on condition type
                        if cond_type == "angle_value":
                            points = fc.get("points", [])
                            expected_value = fc.get("value", "N/A")
                            if points and len(points) > 0 and len(points[0]) == 3:
                                p1, vertex, p3 = points[0]
                                history_parts.append(
                                    f"     💡 Fix: Use rotation to create angle - `rotate : {p1} {expected_value}° {vertex} -> {p3}`"
                                )

                        elif cond_type == "parallel":
                            objects = fc.get("objects", [])
                            if len(objects) == 2:
                                line1, line2 = objects
                                history_parts.append(
                                    f"     💡 Fix: Construct {line2[0]}{line2[1]} parallel to {line1[0]}{line1[1]} using `parallel_line`"
                                )

                        elif cond_type == "perpendicular":
                            objects = fc.get("objects", [])
                            if len(objects) == 2:
                                line1, line2 = objects
                                history_parts.append(
                                    f"     💡 Fix: Use `orthogonal_line : {line2[0]} line_{line1[0]}{line1[1]} -> perp_line`"
                                )

                        elif cond_type == "point_on_segment":
                            point = fc.get("point", "P")
                            segment = fc.get("segment", [])
                            if segment:
                                history_parts.append(
                                    f"     💡 Fix: Place {point} between {segment[0]} and {segment[1]} (use coordinates or intersection)"
                                )

                        elif cond_type == "segment_equality":
                            history_parts.append(
                                f"     💡 Fix: Use same distance for both segments or construct them symmetrically"
                            )

                        elif cond_type == "angle_bisector":
                            angle_points = fc.get("angle_points", [])
                            if len(angle_points) == 3:
                                history_parts.append(
                                    f"     💡 Fix: Use `angular_bisector : {angle_points[0]} {angle_points[1]} {angle_points[2]} -> bisector`"
                                )

                        elif cond_type == "collinear":
                            history_parts.append(
                                f"     💡 Fix: Place all points on the same line"
                            )

                        elif cond_type == "not_collinear":
                            history_parts.append(
                                f"     💡 Fix: Ensure points form a valid triangle, not a line"
                            )

                        elif cond_type == "midpoint_of":
                            point = fc.get("point", "M")
                            segment = fc.get("segment", [])
                            if segment:
                                history_parts.append(
                                    f"     💡 Fix: Use `midpoint : {segment[0]} {segment[1]} -> {point}`"
                                )

                        elif cond_type == "point_on_circle":
                            point = fc.get("point")
                            circle = fc.get("circle")
                            if point and circle:
                                history_parts.append(
                                    f"     💡 Fix: Place {point} on {circle} using rotation or intersection"
                                )

                        elif cond_type == "same_side":
                            points = fc.get("points", [])
                            line = fc.get("line", [])
                            if points and line:
                                history_parts.append(
                                    f"     💡 Fix: Ensure points {', '.join(str(p) for p in points)} are on the same side of the line"
                                )

                        else:
                            # Generic fix suggestion
                            history_parts.append(
                                f"     💡 Review the condition and adjust your construction accordingly"
                            )

            history_parts.append("\n" + "=" * 70 + "\n")

        return "\n".join(history_parts)

    def _validate_solution(
        self, dsl_code: str, problem: BenchmarkProblem
    ) -> Dict[str, Any]:
        """Validate solution against problem requirements."""
        # Create temporary DSL file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(dsl_code)
            dsl_file = f.name

        try:
            validation = self.validator.validate(dsl_file, problem)
            # print(validation)
            # Extract detailed messages from validation details
            failed_conditions_with_messages = []
            if validation.details and "condition_details" in validation.details:
                for detail in validation.details["condition_details"]:
                    if not detail.get("passed", False):
                        condition = detail.get("condition", {})
                        message = detail.get("message", "No message")

                        # Combine condition info with the validation message
                        failed_cond = dict(condition)
                        failed_cond["validation_message"] = message
                        # Include error_type if present
                        if "error_type" in detail:
                            failed_cond["error_type"] = detail["error_type"]
                        failed_conditions_with_messages.append(failed_cond)
            else:
                # Fallback to basic failed conditions
                failed_conditions_with_messages = [
                    c for c in validation.failed_conditions
                ]

            return {
                "success": validation.success,
                "total_score": validation.total_score,
                "object_score": validation.object_score,
                "condition_score": validation.condition_score,
                "missing_objects": validation.missing_objects,
                "failed_conditions": failed_conditions_with_messages,
                "has_dataset_error": validation.has_dataset_error,
                "dataset_error_types": validation.dataset_error_types,
                "details": validation.details,  # Include full validation details
            }
        finally:
            if os.path.exists(dsl_file):
                os.remove(dsl_file)


# Test function
if __name__ == "__main__":
    from src.benchmark.benchmark_dataset import (
        BenchmarkDataset,
        RequiredObjects,
        ConditionBuilder,
        VerificationCondition,
    )

    print("=" * 70)
    print("ReAct Agent Test")
    print("=" * 70)
    print()

    # Create a simple test problem
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]],
    )

    conditions = [
        VerificationCondition.from_dict(ConditionBuilder.not_collinear(["A", "B", "C"]))
    ]

    from src.benchmark.benchmark_dataset import BenchmarkProblem

    problem = BenchmarkProblem(
        id="test_triangle",
        subject="Create a triangle with vertices A, B, and C",
        required_objects=required_objects,
        verification_conditions=conditions,
    )

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Set it in .env file to run the agent.")
        print("For now, skipping agent test.")
    else:
        print("Testing ReAct Agent...")
        agent = ReActAgent(model="gpt-4o", max_iterations=3, verbose=True)

        results = agent.solve(problem)

        print("\n" + "=" * 70)
        print("Results:")
        print(f"  Success: {results['success']}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Memory saved: {results.get('memory_path')}")
