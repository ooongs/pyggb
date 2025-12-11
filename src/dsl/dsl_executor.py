#!/usr/bin/env python3
"""
DSL Executor for ReAct Agent
Safe execution of DSL code with image rendering and error capture.
"""

import os
import io
import sys
import traceback
import tempfile
import base64
from typing import Dict, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.core.random_constr import Construction
from src.utils import get_output_dir


class DSLExecutionResult:
    """Result of DSL execution."""
    
    def __init__(self, success: bool, image_base64: Optional[str] = None,
                 image_path: Optional[str] = None, error: Optional[str] = None,
                 construction: Optional[Construction] = None,
                 stdout: str = "", stderr: str = ""):
        self.success = success
        self.image_base64 = image_base64
        self.image_path = image_path
        self.error = error
        self.construction = construction
        self.stdout = stdout
        self.stderr = stderr
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "has_image": self.image_base64 is not None,
            "image_path": self.image_path,
            "error": self.error,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "num_elements": len(self.construction.elements) if self.construction else 0
        }


class DSLExecutor:
    """Execute DSL code safely and render images."""
    
    def __init__(self, timeout: int = 30, save_images: bool = False, 
                 image_dir: Optional[str] = None, show_labels: bool = True):
        """
        Initialize DSL executor.
        
        Args:
            timeout: Maximum execution time in seconds
            save_images: Whether to save images to disk
            image_dir: Directory to save images (default: agent_images in project root)
            show_labels: Whether to show labels on objects
        """
        self.timeout = timeout
        self.save_images = save_images
        self.image_dir = image_dir or str(get_output_dir("agent_images"))
        self.show_labels = show_labels
        
        if save_images:
            os.makedirs(self.image_dir, exist_ok=True)
    
    def execute(self, dsl_code: str, problem_id: str = "test",
                iteration: int = 0) -> DSLExecutionResult:
        """
        Execute DSL code and return results with image.
        
        Args:
            dsl_code: DSL code to execute
            problem_id: Problem identifier for image naming
            iteration: Iteration number for image naming
            
        Returns:
            DSLExecutionResult with success status, image, and errors
        """
        # Clean DSL code - remove empty lines and normalize whitespace
        lines = []
        for line in dsl_code.split('\n'):
            line = line.strip()
            
            # Remove inline comments (# or //)
            if '#' in line:
                line = line.split('#')[0].strip()
            if '//' in line:
                line = line.split('//')[0].strip()
            
            # Skip empty lines and pure comment lines
            if line and not line.startswith('#') and not line.startswith('//'):
                lines.append(line)
        
        cleaned_dsl = '\n'.join(lines)
        
        # Create temporary DSL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(cleaned_dsl)
            dsl_file = f.name
        
        try:
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            construction = None
            error_message = None
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # Load and execute DSL
                    construction = Construction()
                    construction.load(dsl_file)
                    construction.generate(require_theorem=False, max_attempts=10)
                    
                except Exception as e:
                    # Check if error message contains line information
                    error_str = str(e)
                    
                    if "Error at line" in error_str:
                        # Enhanced error with line information from random_constr
                        error_message = error_str
                    else:
                        # Fallback: Include DSL code snippet in error for debugging
                        error_lines = dsl_code.split('\n')
                        dsl_preview = '\n'.join(error_lines[:10])  # First 10 lines
                        error_message = (
                            f"{type(e).__name__}: {str(e)}\n"
                            f"DSL Preview:\n{dsl_preview}\n"
                            f"Full traceback:\n{traceback.format_exc()}"
                        )
            
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            # If construction succeeded, render image
            if construction and error_message is None:
                try:
                    image_base64, image_path = self._render_image(
                        construction, problem_id, iteration
                    )
                    
                    return DSLExecutionResult(
                        success=True,
                        image_base64=image_base64,
                        image_path=image_path,
                        construction=construction,
                        stdout=stdout_text,
                        stderr=stderr_text
                    )
                except Exception as e:
                    error_message = f"Rendering error: {type(e).__name__}: {str(e)}"
            
            # Execution failed
            return DSLExecutionResult(
                success=False,
                error=error_message or "Unknown error",
                construction=construction,
                stdout=stdout_text,
                stderr=stderr_text
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(dsl_file):
                os.remove(dsl_file)
    
    def _render_image(self, construction: Construction, 
                     problem_id: str, iteration: int) -> Tuple[str, Optional[str]]:
        """
        Render construction to image.
        
        Args:
            construction: Construction object to render
            problem_id: Problem identifier
            iteration: Iteration number
            
        Returns:
            Tuple of (base64_encoded_image, file_path)
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Render construction with labels
        construction.render(ax, show_labels=self.show_labels)
        
        # Save to buffer for base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        
        # Optionally save to file
        image_path = None
        if self.save_images:
            image_path = os.path.join(
                self.image_dir, 
                f"{problem_id}_iter{iteration}.png"
            )
            plt.savefig(image_path, bbox_inches='tight', dpi=100)
        
        plt.close(fig)
        
        return image_base64, image_path
    
    def validate_dsl_syntax(self, dsl_code: str) -> Tuple[bool, Optional[str]]:
        """
        Quick syntax validation without execution.
        
        Args:
            dsl_code: DSL code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        lines = dsl_code.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check basic syntax
            if ':' not in line and 'const' not in line:
                return False, f"Line {i}: Invalid syntax (missing ':')"
            
            if '->' not in line:
                return False, f"Line {i}: Invalid syntax (missing '->')"
            
            # Check for balanced structure
            parts = line.split('->')
            if len(parts) != 2:
                return False, f"Line {i}: Invalid syntax (multiple '->')"
        
        return True, None


# Test function
if __name__ == "__main__":
    executor = DSLExecutor(save_images=True, image_dir="test_images")
    
    # Test DSL with numeric literals
    test_dsl = """
# Define the points of triangle ABC
point : 0 0 -> A
point : 200 0 -> B
rotate : B 80 A -> C

# Create the triangle ABC
polygon : A B C -> triangle_ABC c a b

# Define line BC
line : B C -> line_BC

# Define points D and E on lines AB and AC
point : 100 0 -> D
line : A C -> line_AC
line : A B -> line_AB
intersect : line_AC line_AB -> E

# Create line DE parallel to BC
line : D E -> line_DE
rotate : D 180 line_BC -> D_parallel
line : D_parallel E -> line_DE_parallel
"""
    
    print("Testing DSL Executor...")
    print("="*70)
    
    result = executor.execute(test_dsl, problem_id="test", iteration=0)
    
    print(f"Success: {result.success}")
    print(f"Has image: {result.image_base64 is not None}")
    print(f"Image saved: {result.image_path}")
    print(f"Error: {result.error}")
    print(f"Elements: {len(result.construction.elements) if result.construction else 0}")
    
    if result.success:
        print("\n✓ DSL execution successful!")
    else:
        print(f"\n✗ DSL execution failed: {result.error}")

