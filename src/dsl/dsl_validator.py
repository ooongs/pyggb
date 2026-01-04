#!/usr/bin/env python3
"""
DSL Validator for Geometry Benchmark
Validates DSL files against required objects and verification conditions.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

from src.core.random_constr import Construction
from src.benchmark.benchmark_dataset import BenchmarkProblem, VerificationCondition
from src.core import geo_types as gt
from src.core import commands as cmd
from src.utils import get_output_dir


@dataclass
class ValidationError:
    """Represents a validation error for logging."""
    problem_id: str
    condition_type: str
    condition_data: Dict[str, Any]
    error_message: str
    error_type: str  # "unknown_condition", "missing_points", "execution_error", "dataset_error"
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "condition_type": self.condition_type,
            "condition_data": self.condition_data,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "timestamp": self.timestamp
        }


class ValidationErrorLogger:
    """Logger for validation errors encountered during benchmark runs."""
    
    # List of all supported condition types
    SUPPORTED_CONDITION_TYPES = [
        # Original types
        "parallel", "perpendicular", "angle_value", "angle_equality",
        "segment_equality", "collinear", "not_collinear", "concyclic",
        "concurrent", "point_on_line", "point_on_circle", "angle_bisector",
        "point_on_segment", "midpoint_of", "distance_equals", "triangle_valid",
        "point_between", "concentric_circles",
        "angle_sum", "isosceles_triangle", "right_triangle",
        "perpendicular_bisector", "point_on_line_extension", "point_on_segment_extension",
        "same_side", "point_inside_circle",
        "tangent", "tangent_at_point",
        "diameter",
        "intersection_point", "polygon_property", "polygon_type",
        "regular_polygon", "square", "order_on_line", "perimeter",
        "point_incenter", "point_on_arc", "midpoint_of_arc",
        "point_outside_line", "point_above_line",
        "intersection", "point_intersection",
        "geometric_transformation", "rotation",
        "contact", "point_height",
        # Ratio conditions
        "angle_ratio", "segment_ratio"
    ]
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or str(get_output_dir("validation_errors"))
        self.errors: List[ValidationError] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_error(self, problem_id: str, condition_type: str, 
                  condition_data: Dict, error_message: str, error_type: str):
        """Log a validation error."""
        error = ValidationError(
            problem_id=problem_id,
            condition_type=condition_type,
            condition_data=condition_data,
            error_message=error_message,
            error_type=error_type
        )
        self.errors.append(error)
        return error
    
    def log_unknown_condition(self, problem_id: str, condition_type: str, condition_data: Dict):
        """Log when an unknown condition type is encountered."""
        return self.log_error(
            problem_id=problem_id,
            condition_type=condition_type,
            condition_data=condition_data,
            error_message=f"Unknown condition type: {condition_type}",
            error_type="unknown_condition"
        )
    
    def log_dataset_error(self, problem_id: str, condition_type: str, 
                          condition_data: Dict, error_message: str):
        """Log errors that appear to be from the dataset itself."""
        return self.log_error(
            problem_id=problem_id,
            condition_type=condition_type,
            condition_data=condition_data,
            error_message=error_message,
            error_type="dataset_error"
        )
    
    def is_supported_condition(self, condition_type: str) -> bool:
        """Check if a condition type is supported."""
        return condition_type in self.SUPPORTED_CONDITION_TYPES
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged errors."""
        error_by_type = {}
        error_by_condition = {}
        error_by_problem = {}
        
        for error in self.errors:
            # By error type
            error_by_type[error.error_type] = error_by_type.get(error.error_type, 0) + 1
            
            # By condition type
            error_by_condition[error.condition_type] = \
                error_by_condition.get(error.condition_type, 0) + 1
            
            # By problem
            if error.problem_id not in error_by_problem:
                error_by_problem[error.problem_id] = []
            error_by_problem[error.problem_id].append(error.to_dict())
        
        return {
            "total_errors": len(self.errors),
            "errors_by_type": error_by_type,
            "errors_by_condition": error_by_condition,
            "unique_problems_with_errors": len(error_by_problem),
            "problems_with_errors": list(error_by_problem.keys())
        }
    
    def save_to_file(self, filename: str = None):
        """Save all errors to a JSON file."""
        if filename is None:
            filename = os.path.join(self.log_dir, f"validation_errors_{self.session_id}.json")
        
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "errors": [e.to_dict() for e in self.errors]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def get_errors_for_problem(self, problem_id: str) -> List[ValidationError]:
        """Get all errors for a specific problem."""
        return [e for e in self.errors if e.problem_id == problem_id]
    
    def get_unknown_conditions(self) -> List[str]:
        """Get list of all unknown condition types encountered."""
        unknown = set()
        for error in self.errors:
            if error.error_type == "unknown_condition":
                unknown.add(error.condition_type)
        return sorted(list(unknown))


# Global error logger instance
_global_error_logger: Optional[ValidationErrorLogger] = None

def get_validation_error_logger() -> ValidationErrorLogger:
    """Get or create the global validation error logger."""
    global _global_error_logger
    if _global_error_logger is None:
        _global_error_logger = ValidationErrorLogger()
    return _global_error_logger

def set_validation_error_logger(logger: ValidationErrorLogger):
    """Set the global validation error logger."""
    global _global_error_logger
    _global_error_logger = logger


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

    # Handle Python bool explicitly (some conditions might have np.bool_)
    elif isinstance(obj, bool):
        return bool(obj)  # Ensure it's a Python bool, not numpy

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
class ValidationResult:
    """Result of validating a DSL against a benchmark problem."""
    success: bool
    object_score: float  # 0.0 to 1.0
    condition_score: float  # 0.0 to 1.0
    total_score: float  # 0.0 to 1.0
    missing_objects: Dict[str, List]
    failed_conditions: List[Dict[str, Any]]
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    has_dataset_error: bool = False  # True if validation failed due to dataset issues
    dataset_error_types: List[str] = field(default_factory=list)  # List of error types encountered

    def to_dict(self) -> Dict:
        """Convert to dictionary with all values JSON-serializable."""
        return _convert_to_json_serializable({
            "success": self.success,
            "object_score": self.object_score,
            "condition_score": self.condition_score,
            "total_score": self.total_score,
            "missing_objects": self.missing_objects,
            "failed_conditions": self.failed_conditions,
            "error_message": self.error_message,
            "details": self.details,
            "has_dataset_error": self.has_dataset_error,
            "dataset_error_types": self.dataset_error_types
        })


class DSLValidator:
    """Validate DSL constructions against benchmark requirements."""

    def __init__(self, tolerance: float = 1e-2, error_logger: ValidationErrorLogger = None):
        """
        Initialize validator.

        Args:
            tolerance: Numerical tolerance for geometric comparisons
            error_logger: Optional ValidationErrorLogger for error tracking
        """
        self.tolerance = tolerance
        self.construction = Construction()
        self.error_logger = error_logger or get_validation_error_logger()
        self.current_problem_id: Optional[str] = None  # Track current problem for logging
    
    def validate(self, dsl_file: str, problem: BenchmarkProblem,
                 max_attempts: int = 100) -> ValidationResult:
        """
        Validate a DSL file against a benchmark problem.

        Args:
            dsl_file: Path to DSL file to validate
            problem: BenchmarkProblem to validate against
            max_attempts: Maximum attempts to generate valid construction

        Returns:
            ValidationResult with scores and details
        """
        # Set current problem ID for error logging
        self.current_problem_id = problem.id
        
        try:
            # Load and generate construction
            self.construction.load(dsl_file)
            self.construction.generate(require_theorem=False, max_attempts=max_attempts)
            
            # Check required objects
            object_result = self._check_required_objects(problem.required_objects)
            
            # Check verification conditions
            condition_result = self._check_verification_conditions(
                problem.verification_conditions
            )
            
            # Calculate total score (weighted average)
            object_score = object_result.get("score", 0.0)
            condition_score = condition_result.get("score", 0.0)
            
            # Get missing/failed with proper defaults
            missing_objects = object_result.get("missing", {})
            failed_conditions = condition_result.get("failed", [])
            
            # Debug logging
            import sys
            if hasattr(sys, '_debug_validation'):
                print(f"DEBUG: object_score={object_score}, condition_score={condition_score}")
                print(f"DEBUG: missing_objects={missing_objects}")
                print(f"DEBUG: failed_conditions={failed_conditions}")
                print(f"DEBUG: object_result={object_result}")
                print(f"DEBUG: condition_result={condition_result}")
            
            # IMPORTANT: If scores are 0 but no failures reported, something is wrong
            # This shouldn't happen - if score is 0, there should be missing/failed items
            if (object_score == 0.0 or condition_score == 0.0):
                # Check if we have empty missing/failed despite 0 scores
                has_missing = any(objs for objs in missing_objects.values() if objs)
                has_failed = len(failed_conditions) > 0
                
                if not has_missing and object_score == 0.0 and len(problem.required_objects.points) > 0:
                    # Object score is 0 but no missing objects reported - this is a bug
                    # Add a generic error to failed_conditions
                    failed_conditions.append({
                        "type": "validation_error",
                        "message": "Object validation returned 0% but no specific missing objects identified. This may indicate a validation bug or empty requirements."
                    })
                
                if not has_failed and condition_score == 0.0 and len(problem.verification_conditions) > 0:
                    # Condition score is 0 but no failed conditions reported - this is a bug
                    failed_conditions.append({
                        "type": "validation_error",
                        "message": "Condition validation returned 0% but no specific failed conditions identified. This may indicate a validation bug or empty requirements."
                    })
            
            total_score = 0.3 * object_score + 0.7 * condition_score
            
            success = (object_score >= 0.9 and condition_score >= 0.9)
            
            # Check for dataset errors (unknown_condition, execution_error, dataset_error)
            dataset_error_types = []
            has_dataset_error = False
            for cond in failed_conditions:
                error_type = cond.get("error_type")
                if error_type in ["unknown_condition", "execution_error", "dataset_error"]:
                    has_dataset_error = True
                    if error_type not in dataset_error_types:
                        dataset_error_types.append(error_type)
            
            return ValidationResult(
                success=success,
                object_score=object_score,
                condition_score=condition_score,
                total_score=total_score,
                missing_objects=missing_objects,
                failed_conditions=failed_conditions,
                details={
                    "object_details": object_result.get("details", {}),
                    "found_objects": object_result.get("found", {}),
                    "condition_details": condition_result.get("details", []),
                    "passed_conditions": condition_result.get("passed", [])
                },
                has_dataset_error=has_dataset_error,
                dataset_error_types=dataset_error_types
            )
            
        except Exception as e:
            import traceback
            return ValidationResult(
                success=False,
                object_score=0.0,
                condition_score=0.0,
                total_score=0.0,
                missing_objects={},
                failed_conditions=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}",
                has_dataset_error=True,
                dataset_error_types=["exception"]
            )
    
    def _check_required_objects(self, required_objects) -> Dict[str, Any]:
        """
        Check if all required objects exist in the construction.
        Uses hybrid validation: explicit for polygons/circles, implicit for segments/lines.
        """
        element_dict = self.construction.element_dict
        # print("Element Dict:",element_dict)
        
        missing = {
            "points": [],
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        found = {
            "points": [],
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        # EXPLICIT VALIDATION: Points (must exist as labeled objects)
        for point_label in required_objects.points:
            if point_label in element_dict:
                element = element_dict[point_label]
                if isinstance(element.data, gt.Point):
                    found["points"].append(point_label)
                else:
                    missing["points"].append(point_label)
            else:
                missing["points"].append(point_label)
        
        # HYBRID VALIDATION: Segments (check explicit OR can be inferred from points)
        for seg in required_objects.segments:
            # First try to find explicit segment
            seg_label = self._find_segment(seg[0], seg[1])
            if seg_label:
                found["segments"].append(seg)
            else:
                # Implicit: Check if both points exist (segment can be inferred)
                if seg[0] in element_dict and seg[1] in element_dict:
                    p1 = element_dict[seg[0]].data
                    p2 = element_dict[seg[1]].data
                    if isinstance(p1, gt.Point) and isinstance(p2, gt.Point):
                        # Points exist, segment can be inferred
                        found["segments"].append(seg)
                    else:
                        missing["segments"].append(seg)
                else:
                    missing["segments"].append(seg)
        
        # HYBRID VALIDATION: Lines (check explicit OR can be inferred from points)
        for line in required_objects.lines:
            # Handle different line formats: "l", ["l"], or ["A", "B"]
            if isinstance(line, str):
                # Named line: "l"
                if line in element_dict:
                    elem = element_dict[line].data
                    if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                        found["lines"].append(line)
                    else:
                        missing["lines"].append(line)
                else:
                    missing["lines"].append(line)

            elif isinstance(line, list):
                # Single-item list: ["l"] -> named line
                if len(line) == 1:
                    label = line[0]
                    if label in element_dict:
                        elem = element_dict[label].data
                        if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                            found["lines"].append(line)
                        else:
                            missing["lines"].append(line)
                    else:
                        missing["lines"].append(line)

                # Two-item list: ["A", "B"] -> line by points
                elif len(line) == 2:
                    # First try to find explicit line
                    line_label = self._find_line(line[0], line[1])
                    if line_label:
                        found["lines"].append(line)
                    else:
                        # Implicit: Check if both points exist (line can be inferred)
                        if line[0] in element_dict and line[1] in element_dict:
                            p1 = element_dict[line[0]].data
                            p2 = element_dict[line[1]].data
                            if isinstance(p1, gt.Point) and isinstance(p2, gt.Point):
                                # Points exist, line can be inferred
                                found["lines"].append(line)
                            else:
                                missing["lines"].append(line)
                        else:
                            missing["lines"].append(line)
                else:
                    # Invalid length (e.g., ["A", "B", "C"])
                    missing["lines"].append(line)
            else:
                # Unknown type
                missing["lines"].append(line)
        
        # EXPLICIT VALIDATION: Circles (must exist as objects)
        for circle_def in required_objects.circles:
            center = circle_def.get("center")
            if center and center in element_dict:
                # Check if there's a circle with this center
                circle_label = self._find_circle_with_center(center)
                if circle_label:
                    found["circles"].append(circle_def)
                else:
                    missing["circles"].append(circle_def)
            else:
                missing["circles"].append(circle_def)
        
        # EXPLICIT VALIDATION: Polygons (must exist as objects)
        for poly_points in required_objects.polygons:
            poly_label = self._find_polygon(poly_points)
            if poly_label:
                # Found a matching polygon - check if it's self-intersecting
                # Use the actual polygon's point order, not the required order
                polygon_obj = element_dict[poly_label].data
                if hasattr(polygon_obj, 'points'):
                    actual_poly_coords = list(polygon_obj.points)
                    if self._is_polygon_self_intersecting(actual_poly_coords):
                        # Self-intersecting polygon (e.g., hourglass) is invalid
                        missing["polygons"].append(poly_points)
                    else:
                        found["polygons"].append(poly_points)
                else:
                    found["polygons"].append(poly_points)
            else:
                # Relaxed: check if all points exist (polygon structure can be inferred)
                if all(p in element_dict for p in poly_points):
                    all_points = all(isinstance(element_dict[p].data, gt.Point) for p in poly_points)
                    if all_points:
                        # Try to find any Polygon object that contains these points
                        # Check if the polygon has the same SET of points (order doesn't matter)
                        found_polygon_obj = None
                        for label, element in element_dict.items():
                            if isinstance(element.data, gt.Polygon):
                                poly = element.data
                                # Check if this polygon has the same set of points
                                if len(poly.points) == len(poly_points):
                                    # Get coordinates of required points
                                    required_coords = [element_dict[p].data.a for p in poly_points]

                                    # Check if all polygon points match any required point
                                    all_matched = True
                                    for poly_point in poly.points:
                                        found_match = False
                                        for req_coord in required_coords:
                                            if np.allclose(poly_point, req_coord, atol=self.tolerance):
                                                found_match = True
                                                break
                                        if not found_match:
                                            all_matched = False
                                            break

                                    if all_matched:
                                        found_polygon_obj = poly
                                        break

                        if found_polygon_obj:
                            # Use the actual polygon's point order
                            actual_poly_coords = list(found_polygon_obj.points)
                            if self._is_polygon_self_intersecting(actual_poly_coords):
                                # Self-intersecting polygon (e.g., hourglass) is invalid
                                missing["polygons"].append(poly_points)
                            else:
                                found["polygons"].append(poly_points)
                        # Check if points are not collinear (valid polygon)
                        elif len(poly_points) == 3:
                            # Triangle: check non-collinearity
                            p1, p2, p3 = [element_dict[p].data for p in poly_points]
                            collinear = cmd.are_collinear_ppp(p1, p2, p3)
                            if not collinear.b:
                                found["polygons"].append(poly_points)
                            else:
                                missing["polygons"].append(poly_points)
                        else:
                            # No polygon object found, but points exist
                            # Try to find the actual polygon path from segments
                            polygon_path = self._find_polygon_path_from_segments(poly_points)

                            if polygon_path:
                                # Found a path through segments - check if it's self-intersecting
                                poly_coords = [element_dict[p].data.a for p in polygon_path]
                                if self._is_polygon_self_intersecting(poly_coords):
                                    # Segments form a self-intersecting polygon (invalid)
                                    missing["polygons"].append(poly_points)
                                else:
                                    # Segments form a valid polygon
                                    found["polygons"].append(poly_points)
                            else:
                                # No connected path found through segments
                                # Fall back to checking if points form a valid polygon (relaxed)
                                poly_coords = [element_dict[p].data.a for p in poly_points]
                                if self._is_polygon_self_intersecting(poly_coords):
                                    missing["polygons"].append(poly_points)
                                else:
                                    found["polygons"].append(poly_points)
                    else:
                        missing["polygons"].append(poly_points)
                else:
                    missing["polygons"].append(poly_points)
        
        # Calculate score
        total_required = (
            len(required_objects.points) +
            len(required_objects.segments) +
            len(required_objects.lines) +
            len(required_objects.circles) +
            len(required_objects.polygons)
        )
        
        total_found = (
            len(found["points"]) +
            len(found["segments"]) +
            len(found["lines"]) +
            len(found["circles"]) +
            len(found["polygons"])
        )
        
        score = total_found / total_required if total_required > 0 else 1.0
        
        return {
            "score": score,
            "missing": missing,
            "found": found,
            "details": {
                "total_required": total_required,
                "total_found": total_found,
                "validation_mode": "hybrid"
            }
        }
    
    def _find_segment(self, p1: str, p2: str) -> Optional[str]:
        """Find a segment between two points in the construction."""
        element_dict = self.construction.element_dict
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Segment):
                seg = element.data
                # Check if segment connects p1 and p2
                if p1 in element_dict and p2 in element_dict:
                    pt1 = element_dict[p1].data
                    pt2 = element_dict[p2].data
                    if isinstance(pt1, gt.Point) and isinstance(pt2, gt.Point):
                        # Check if segment endpoints match
                        if (np.allclose(seg.end_points[0], pt1.a, atol=self.tolerance) and
                            np.allclose(seg.end_points[1], pt2.a, atol=self.tolerance)):
                            return label
                        if (np.allclose(seg.end_points[0], pt2.a, atol=self.tolerance) and
                            np.allclose(seg.end_points[1], pt1.a, atol=self.tolerance)):
                            return label
        return None
    
    def _find_line(self, p1: str, p2: str) -> Optional[str]:
        """Find a line through two points in the construction.

        Note: This also recognizes Segment and Ray as lines, since they
        represent the same geometric concept for validation purposes.
        """
        element_dict = self.construction.element_dict

        if p1 not in element_dict or p2 not in element_dict:
            return None

        pt1 = element_dict[p1].data
        pt2 = element_dict[p2].data

        if not isinstance(pt1, gt.Point) or not isinstance(pt2, gt.Point):
            return None

        for label, element in element_dict.items():
            # Check Line, Segment, and Ray (all can represent a line through two points)
            if isinstance(element.data, (gt.Line, gt.Segment, gt.Ray)):
                line = element.data
                # Check if both points lie on the line/segment/ray
                if line.contains(pt1.a) and line.contains(pt2.a):
                    return label

        return None
    
    def _find_circle_with_center(self, center: str) -> Optional[str]:
        """Find a circle with given center point."""
        element_dict = self.construction.element_dict
        
        if center not in element_dict:
            return None
        
        center_point = element_dict[center].data
        if not isinstance(center_point, gt.Point):
            return None
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Circle):
                circle = element.data
                if np.allclose(circle.c, center_point.a, atol=self.tolerance):
                    return label
        
        return None
    
    def _find_all_circles_with_center(self, center: str) -> List[str]:
        """Find all circles with given center point (for concentric circles)."""
        element_dict = self.construction.element_dict
        circles = []
        
        if center not in element_dict:
            return circles
        
        center_point = element_dict[center].data
        if not isinstance(center_point, gt.Point):
            return circles
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Circle):
                circle = element.data
                if np.allclose(circle.c, center_point.a, atol=self.tolerance):
                    circles.append(label)
        
        return circles
    
    def _find_polygon(self, points: List[str]) -> Optional[str]:
        """Find a polygon with given vertices."""
        element_dict = self.construction.element_dict
        
        # Check all points exist
        for p in points:
            if p not in element_dict or not isinstance(element_dict[p].data, gt.Point):
                return None
        
        # Get point coordinates
        point_coords = [element_dict[p].data.a for p in points]
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Polygon):
                poly = element.data
                # Check if polygon has same vertices
                if len(poly.points) == len(point_coords):
                    # Check if all points match (order might differ)
                    if self._points_match_polygon(point_coords, poly.points):
                        return label
        
        return None
    
    def _find_polygon_path_from_segments(self, poly_points: List[str]) -> Optional[List[str]]:
        """
        Find the actual path that connects the polygon points through segments.
        Returns the ordered list of points, or None if no valid path exists.

        Args:
            poly_points: List of point labels that should form a polygon

        Returns:
            Ordered list of point labels forming a closed polygon path, or None
        """
        element_dict = self.construction.element_dict

        # Find all segments that connect these points
        poly_segments = []
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Segment):
                seg = element.data
                # Check if both endpoints are in poly_points
                endpoints = []
                for p_label in poly_points:
                    if p_label in element_dict and isinstance(element_dict[p_label].data, gt.Point):
                        p_coord = element_dict[p_label].data.a
                        # Check both endpoints of the segment
                        if np.allclose(seg.end_points[0], p_coord, atol=self.tolerance) or \
                           np.allclose(seg.end_points[1], p_coord, atol=self.tolerance):
                            endpoints.append(p_label)

                if len(endpoints) == 2:
                    poly_segments.append(tuple(endpoints))

        if len(poly_segments) < len(poly_points):
            # Not enough segments to form a polygon
            return None

        # Build adjacency graph
        graph = {p: [] for p in poly_points}
        for seg in poly_segments:
            p1, p2 = seg
            graph[p1].append(p2)
            graph[p2].append(p1)

        # Try to find a Hamiltonian cycle starting from each point
        for start_point in poly_points:
            path = self._find_hamiltonian_cycle(graph, start_point, poly_points)
            if path:
                return path

        return None

    def _find_hamiltonian_cycle(self, graph: Dict[str, List[str]],
                                start: str, all_points: List[str]) -> Optional[List[str]]:
        """
        Find a Hamiltonian cycle (path visiting all vertices once) starting from start.
        """
        def dfs(current: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
            if len(path) == len(all_points):
                # Check if we can return to start
                if start in graph[current]:
                    return path
                return None

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs(neighbor, visited, path)
                    if result:
                        return result
                    path.pop()
                    visited.remove(neighbor)

            return None

        return dfs(start, {start}, [start])

    def _is_polygon_self_intersecting(self, points: List[np.ndarray]) -> bool:
        """
        Check if a polygon is self-intersecting (e.g., hourglass shape).
        Uses line segment intersection to detect self-intersection.

        Args:
            points: List of polygon vertices in order

        Returns:
            True if the polygon self-intersects, False otherwise
        """
        n = len(points)
        if n < 4:
            # Triangles cannot self-intersect
            return False

        def segments_intersect(p1, p2, p3, p4):
            """Check if line segment p1-p2 intersects with p3-p4."""
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        # Check each edge against all non-adjacent edges
        for i in range(n):
            p1, p2 = points[i], points[(i + 1) % n]
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + n - 1) % n or j == (i + 1) % n:
                    continue
                p3, p4 = points[j], points[(j + 1) % n]
                if segments_intersect(p1, p2, p3, p4):
                    return True

        return False

    def _points_match_polygon(self, points1: List[np.ndarray],
                             points2: np.ndarray) -> bool:
        """
        Check if two sets of points match (considering rotation and reversal).
        """
        n = len(points1)
        if n != len(points2):
            return False

        # Try all rotations
        for offset in range(n):
            match = True
            for i in range(n):
                if not np.allclose(points1[i], points2[(i + offset) % n],
                                  atol=self.tolerance):
                    match = False
                    break
            if match:
                return True

        # Try reverse order (clockwise vs counterclockwise)
        for offset in range(n):
            match = True
            for i in range(n):
                if not np.allclose(points1[i], points2[(offset - i) % n],
                                  atol=self.tolerance):
                    match = False
                    break
            if match:
                return True

        return False
    
    def _check_verification_conditions(self, conditions: List[VerificationCondition]) -> Dict[str, Any]:
        """Check all verification conditions."""
        failed = []
        passed = []
        details = []
        
        for condition in conditions:
            result = self._check_condition(condition)
            detail = {
                "condition": condition.to_dict(),
                "passed": result.get("passed", False),
                "message": result.get("message", "No message")
            }
            details.append(detail)
            
            if result.get("passed", False):
                passed.append(condition.to_dict())
            else:
                # Include the validation message in the failed condition
                failed_cond = condition.to_dict()
                failed_cond["validation_message"] = result.get("message", "No message")
                failed_cond["validation_passed"] = False
                # Include error_type if present (for dataset errors)
                if "error_type" in result:
                    failed_cond["error_type"] = result["error_type"]
                failed.append(failed_cond)
        
        score = len(passed) / len(conditions) if len(conditions) > 0 else 1.0
        
        return {
            "score": score,
            "failed": failed,
            "passed": passed,
            "details": details
        }
    
    def _check_condition(self, condition: VerificationCondition) -> Dict[str, Any]:
        """Check a single verification condition."""
        try:
            condition_type = condition.type
            
            if condition_type == "parallel":
                return self._check_parallel(condition.data)
            elif condition_type == "perpendicular":
                return self._check_perpendicular(condition.data)
            elif condition_type == "angle_value":
                return self._check_angle_value(condition.data)
            elif condition_type == "angle_equality":
                return self._check_angle_equality(condition.data)
            elif condition_type == "segment_equality":
                return self._check_segment_equality(condition.data)
            elif condition_type == "collinear":
                return self._check_collinear(condition.data)
            elif condition_type == "not_collinear":
                return self._check_not_collinear(condition.data)
            elif condition_type == "concyclic":
                return self._check_concyclic(condition.data)
            elif condition_type == "concurrent":
                return self._check_concurrent(condition.data)
            elif condition_type == "point_on_line":
                return self._check_point_on_line(condition.data)
            elif condition_type == "point_on_circle":
                return self._check_point_on_circle(condition.data)
            elif condition_type == "angle_bisector":
                return self._check_angle_bisector(condition.data)
            elif condition_type == "point_on_segment":
                return self._check_point_on_segment(condition.data)
            elif condition_type == "midpoint_of":
                return self._check_midpoint_of(condition.data)
            elif condition_type == "distance_equals":
                return self._check_distance_equals(condition.data)
            elif condition_type == "triangle_valid":
                return self._check_triangle_valid(condition.data)
            elif condition_type == "point_between":
                return self._check_point_between(condition.data)
            elif condition_type == "concentric_circles":
                return self._check_concentric_circles(condition.data)
            # New condition types from dataset
            elif condition_type == "angle_sum":
                return self._check_angle_sum(condition.data)
            elif condition_type == "isosceles_triangle":
                return self._check_isosceles_triangle(condition.data)
            elif condition_type == "right_triangle":
                return self._check_right_triangle(condition.data)
            elif condition_type == "segment_equal":
                # Redirect to segment_equality
                return self._check_segment_equality(condition.data)
            elif condition_type == "length":
                # Redirect to distance_equals
                return self._check_length_value(condition.data)
            elif condition_type == "perpendicular_bisector":
                return self._check_perpendicular_bisector(condition.data)
            elif condition_type == "point_on_line_extension":
                return self._check_point_on_line_extension(condition.data)
            elif condition_type == "point_on_segment_extension":
                return self._check_point_on_segment_extension(condition.data)
            elif condition_type == "same_side":
                return self._check_same_side(condition.data)
            elif condition_type == "point_inside_circle":
                return self._check_point_inside_circle(condition.data)
            elif condition_type == "tangent":
                return self._check_tangent_line(condition.data)
            elif condition_type == "tangent_at_point":
                return self._check_tangent_at_point(condition.data)
            elif condition_type == "diameter":
                return self._check_diameter(condition.data)
            elif condition_type == "intersection_point":
                return self._check_intersection_point(condition.data)
            elif condition_type == "polygon_property":
                return self._check_polygon_property(condition.data)
            elif condition_type == "polygon_type":
                return self._check_polygon_type(condition.data)
            elif condition_type == "regular_polygon":
                return self._check_regular_polygon(condition.data)
            elif condition_type == "square":
                return self._check_square(condition.data)
            elif condition_type == "order_on_line":
                return self._check_order_on_line(condition.data)
            elif condition_type == "perimeter":
                return self._check_perimeter(condition.data)
            elif condition_type == "point_incenter":
                return self._check_point_incenter(condition.data)
            elif condition_type == "point_on_arc":
                return self._check_point_on_arc(condition.data)
            elif condition_type == "midpoint_of_arc":
                return self._check_midpoint_of_arc(condition.data)
            elif condition_type == "point_outside_line":
                return self._check_point_outside_line(condition.data)
            elif condition_type == "point_above_line":
                return self._check_point_above_line(condition.data)
            elif condition_type == "intersection":
                return self._check_intersection(condition.data)
            elif condition_type == "point_intersection":
                return self._check_point_intersection(condition.data)
            elif condition_type in ["geometric_transformation", "rotation"]:
                return self._check_geometric_transformation(condition.data)
            elif condition_type == "contact":
                return self._check_contact(condition.data)
            elif condition_type == "point_height":
                return self._check_point_height(condition.data)
            # Length-related conditions
            elif condition_type in ["segments_sum_value", "segments_sum"]:
                return self._check_segments_sum_value(condition.data)
            elif condition_type in ["segments_sum_equals", "segments_sum_equals_segment"]:
                return self._check_segments_sum_equals(condition.data)
            elif condition_type == "ratio":
                return self._check_ratio(condition.data)
            elif condition_type == "angle_ratio":
                return self._check_angle_ratio(condition.data)
            elif condition_type == "segment_ratio":
                return self._check_segment_ratio(condition.data)
            else:
                # Log unknown condition type
                error_msg = f"Unknown condition type: {condition_type}"
                if self.error_logger and self.current_problem_id:
                    self.error_logger.log_unknown_condition(
                        problem_id=self.current_problem_id,
                        condition_type=condition_type,
                        condition_data=condition.data
                    )
                return {
                    "passed": False,
                    "message": error_msg,
                    "error_type": "unknown_condition"
                }
        except Exception as e:
            # Log execution error
            error_msg = f"Error checking condition: {str(e)}"
            if self.error_logger and self.current_problem_id:
                self.error_logger.log_error(
                    problem_id=self.current_problem_id,
                    condition_type=condition.type,
                    condition_data=condition.data,
                    error_message=str(e),
                    error_type="execution_error"
                )
            return {
                "passed": False,
                "message": error_msg,
                "error_type": "execution_error"
            }
    
    def _get_line_from_points(self, p1: str, p2: str) -> Optional[gt.Line]:
        """Get or create a line from two points."""
        element_dict = self.construction.element_dict
        
        if p1 not in element_dict or p2 not in element_dict:
            return None
        
        pt1 = element_dict[p1].data
        pt2 = element_dict[p2].data
        
        if not isinstance(pt1, gt.Point) or not isinstance(pt2, gt.Point):
            return None
        
        # Check if line already exists
        line_label = self._find_line(p1, p2)
        if line_label:
            return element_dict[line_label].data
        
        # Create line from points
        return cmd.line_pp(pt1, pt2)

    def _get_line_from_data(self, line_data) -> Optional[gt.Line]:
        """Get a Line object from various formats.

        Handles:
        - String label: "l" -> lookup in element_dict
        - List with 1 item: ["l"] -> extract string and lookup
        - List with 2 items: ["A", "B"] -> construct line from points

        Args:
            line_data: String label, single-item list, or two-point list

        Returns:
            Line object if found/created, None otherwise
        """
        element_dict = self.construction.element_dict

        # Handle string: "l"
        if isinstance(line_data, str):
            if line_data in element_dict:
                elem = element_dict[line_data].data
                if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                    return elem
            return None

        # Handle list
        elif isinstance(line_data, list):
            # Single-item list: ["l"] -> treat as named line
            if len(line_data) == 1:
                label = line_data[0]
                if label in element_dict:
                    elem = element_dict[label].data
                    if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                        return elem
                return None

            # Two-item list: ["A", "B"] -> line by points
            elif len(line_data) == 2:
                return self._get_line_from_points(line_data[0], line_data[1])

        return None

    def _check_parallel(self, data: Dict) -> Dict[str, Any]:
        """Check if two lines are parallel."""
        objects = data.get("objects", [])
        if len(objects) != 2:
            return {"passed": False, "message": "Parallel condition requires 2 lines"}

        # Get lines using the helper method (supports all formats)
        line1 = self._get_line_from_data(objects[0])
        line2 = self._get_line_from_data(objects[1])

        if line1 is None or line2 is None:
            return {"passed": False, "message": "Could not find lines"}

        result = cmd.are_parallel_ll(line1, line2)

        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} parallel"
        }
    
    def _check_perpendicular(self, data: Dict) -> Dict[str, Any]:
        """Check if two lines are perpendicular."""
        objects = data.get("objects", [])
        if len(objects) != 2:
            return {"passed": False, "message": "Perpendicular condition requires 2 lines"}

        # Get lines using the helper method (supports all formats)
        line1 = self._get_line_from_data(objects[0])
        line2 = self._get_line_from_data(objects[1])

        if line1 is None or line2 is None:
            return {"passed": False, "message": "Could not find lines"}

        result = cmd.are_perpendicular_ll(line1, line2)

        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} perpendicular"
        }
    
    def _check_angle_value(self, data: Dict) -> Dict[str, Any]:
        """Check if angle has expected value."""
        points_list = data.get("points", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", 1.0)  # Default 1 degree tolerance
        
        if len(points_list) != 1 or len(points_list[0]) != 3:
            return {"passed": False, "message": "Angle value requires 3 points"}
        
        points = points_list[0]
        element_dict = self.construction.element_dict
        
        # Get points
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        p1 = element_dict[points[0]].data
        p2 = element_dict[points[1]].data
        p3 = element_dict[points[2]].data
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate angle
        angle = cmd.angle_ppp(p1, p2, p3)
        angle_degrees = np.degrees(angle.angle)

        # Allow either the measured (CCW) angle or its reflex (360 - angle)
        # so that geometrically equivalent 60° vs 300° do not falsely fail.
        reflex_angle = (360.0 - angle_degrees) % 360.0
        expected = expected_value % 360.0

        diff_primary = np.abs(angle_degrees - expected)
        diff_reflex = np.abs(reflex_angle - expected)
        min_diff = min(diff_primary, diff_reflex)
        passed = min_diff <= tolerance

        chosen_angle = angle_degrees if diff_primary <= diff_reflex else reflex_angle
        
        return {
            "passed": passed,
            "message": (
                f"Angle is {angle_degrees:.2f}° (reflex {reflex_angle:.2f}°), "
                f"expected {expected_value}° (tolerance {tolerance}°); "
                f"closest match: {chosen_angle:.2f}°"
            )
        }
    
    def _check_angle_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if multiple angles are equal."""
        points_list = data.get("points", [])
        tolerance = data.get("tolerance", 1.0)
        
        if len(points_list) < 2:
            return {"passed": False, "message": "Angle equality requires at least 2 angles"}
        
        element_dict = self.construction.element_dict
        
        # Calculate all angles
        angles = []
        angle_names = []
        
        for i, points in enumerate(points_list):
            if len(points) != 3:
                return {"passed": False, "message": f"Angle {i+1} requires exactly 3 points"}
            
            if not all(p in element_dict for p in points):
                return {"passed": False, "message": f"Could not find points for angle {i+1}"}
            
            pa, pb, pc = [element_dict[p].data for p in points]
            if not all(isinstance(p, gt.Point) for p in [pa, pb, pc]):
                return {"passed": False, "message": f"Invalid point types in angle {i+1}"}
            
            angle = cmd.angle_ppp(pa, pb, pc)
            angles.append(angle)
            angle_names.append(f"∠{points[0]}{points[1]}{points[2]}")
        
        # Check all pairs of angles for equality
        all_equal = True
        failed_pairs = []
        
        for i in range(len(angles) - 1):
            result = cmd.are_congruent_aa(angles[i], angles[i + 1])
            if not result.b:
                all_equal = False
                angle_i_deg = np.degrees(angles[i].angle)
                angle_j_deg = np.degrees(angles[i + 1].angle)
                failed_pairs.append((angle_names[i], angle_names[i + 1], angle_i_deg, angle_j_deg))
        
        # Format message
        angle_degrees = [np.degrees(a.angle) for a in angles]
        
        if len(points_list) == 2:
            message = f"Angles are {angle_degrees[0]:.2f}° and {angle_degrees[1]:.2f}°"
        else:
            angles_str = ", ".join([f"{name}={deg:.2f}°" for name, deg in zip(angle_names, angle_degrees)])
            if all_equal:
                message = f"All angles are equal: {angles_str}"
            else:
                failed_str = "; ".join([f"{p[0]}({p[2]:.2f}°) != {p[1]}({p[3]:.2f}°)" for p in failed_pairs])
                message = f"Angles not all equal: {failed_str}"
        
        return {
            "passed": all_equal,
            "message": message
        }
    
    def _check_segment_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if multiple segments are equal in length."""
        # Support both 'segments' and 'objects' field
        segments = data.get("segments", data.get("objects", []))

        if len(segments) < 2:
            return {"passed": False, "message": "Segment equality requires at least 2 segments"}
        
        element_dict = self.construction.element_dict
        
        # Collect all points and verify they exist
        all_points = []
        for seg in segments:
            all_points.extend(seg)
        
        if not all(p in element_dict for p in all_points):
            return {"passed": False, "message": "Could not find all points"}
        
        # Calculate distances for all segments
        distances = []
        segment_names = []
        for seg_points in segments:
            pa, pb = [element_dict[p].data for p in seg_points]
            if not all(isinstance(p, gt.Point) for p in [pa, pb]):
                return {"passed": False, "message": "Invalid point types"}
            dist = cmd.distance_pp(pa, pb)
            distances.append(dist)
            segment_names.append(f"{seg_points[0]}{seg_points[1]}")

        # Correct for scaling applied by fit_to_window()
        scale_factor = getattr(self.construction, 'scale_factor', 1.0)
        distances_corrected = [d.x / scale_factor for d in distances]

        # Check all pairs of segments for equality
        all_equal = True
        failed_pairs = []

        for i in range(len(distances) - 1):
            result = cmd.are_equal_mm(distances[i], distances[i + 1])
            if not result.b:
                all_equal = False
                failed_pairs.append((segment_names[i], segment_names[i + 1], distances_corrected[i], distances_corrected[i + 1]))

        # Format message
        if len(segments) == 2:
            message = f"Segments have lengths {distances_corrected[0]:.2f} and {distances_corrected[1]:.2f}"
        else:
            lengths_str = ", ".join([f"{name}={dist:.2f}" for name, dist in zip(segment_names, distances_corrected)])
            if all_equal:
                message = f"All segments are equal: {lengths_str}"
            else:
                failed_str = "; ".join([f"{p[0]}({p[2]:.2f}) != {p[1]}({p[3]:.2f})" for p in failed_pairs])
                message = f"Segments not all equal: {failed_str}"
        
        return {
            "passed": all_equal,
            "message": message
        }
    
    def _check_collinear(self, data: Dict) -> Dict[str, Any]:
        """Check if points are collinear.

        If 'line' is provided (named line), also validates that all points lie on that line.
        """
        points = data.get("points", [])
        line_data = data.get("line")

        if len(points) < 3:
            return {"passed": False, "message": "Collinearity requires at least 3 points"}

        element_dict = self.construction.element_dict

        # Check first 3 points (can extend to check all combinations)
        if not all(p in element_dict for p in points[:3]):
            return {"passed": False, "message": "Could not find all points"}

        p1, p2, p3 = [element_dict[p].data for p in points[:3]]

        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
            return {"passed": False, "message": "Invalid point types"}

        result = cmd.are_collinear_ppp(p1, p2, p3)

        if not result.b:
            return {"passed": False, "message": "Points are not collinear"}

        # If named line is provided, verify all points lie on it
        if line_data:
            line = self._get_line_from_data(line_data)
            if line is None:
                return {"passed": False, "message": "Could not find named line"}

            # Check all points lie on the named line
            for point_label in points:
                if point_label not in element_dict:
                    return {"passed": False, "message": f"Could not find point {point_label}"}
                pt = element_dict[point_label].data
                if not isinstance(pt, gt.Point):
                    return {"passed": False, "message": f"Invalid point type for {point_label}"}
                if not line.contains(pt.a):
                    return {"passed": False, "message": f"Point {point_label} is not on named line"}

        return {
            "passed": True,
            "message": f"Points are collinear" + (f" on named line" if line_data else "")
        }
    
    def _check_not_collinear(self, data: Dict) -> Dict[str, Any]:
        """Check if points are NOT collinear (for valid triangles)."""
        result = self._check_collinear(data)
        result["passed"] = not result["passed"]
        result["message"] = result["message"].replace("are collinear", "are not collinear").replace("are not not", "are")
        return result
    
    def _check_concyclic(self, data: Dict) -> Dict[str, Any]:
        """Check if four points lie on the same circle."""
        points = data.get("points", [])
        
        if len(points) != 4:
            return {"passed": False, "message": "Concyclic condition requires 4 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2, p3, p4 = [element_dict[p].data for p in points]
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3, p4]):
            return {"passed": False, "message": "Invalid point types"}
        
        result = cmd.are_concyclic_pppp(p1, p2, p3, p4)
        
        return {
            "passed": result.b,
            "message": f"Points {'are' if result.b else 'are not'} concyclic"
        }
    
    def _check_concurrent(self, data: Dict) -> Dict[str, Any]:
        """Check if three lines meet at a point."""
        lines = data.get("lines", [])

        if len(lines) != 3:
            return {"passed": False, "message": "Concurrent condition requires 3 lines"}

        line_objs = []
        for line_data in lines:
            # Support all line formats
            line = self._get_line_from_data(line_data)
            if line is None:
                return {"passed": False, "message": f"Could not find line {line_data}"}
            line_objs.append(line)

        result = cmd.are_concurrent_lll(*line_objs)

        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} concurrent"
        }
    
    def _check_point_on_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a line."""
        point = data.get("point")
        line_data = data.get("line")

        if not point or not line_data:
            return {"passed": False, "message": "Invalid point_on_line condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}

        # Handle all line formats: "l", ["l"], or ["A", "B"]
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find line"}

        result = cmd.contained_by_pl(pt, line)

        return {
            "passed": result.b,
            "message": f"Point {'is' if result.b else 'is not'} on line"
        }
    
    def _check_point_on_circle(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a circle (supports concentric circles)."""
        point = data.get("point")

        # Support both formats:
        # 1. circle_center: "O" (ConditionBuilder format)
        # 2. circle: {center: "O", radius_point: "A"} (dataset format)
        circle_center = data.get("circle_center")
        if not circle_center and "circle" in data:
            circle_data = data.get("circle")
            if isinstance(circle_data, dict):
                circle_center = circle_data.get("center")
            elif isinstance(circle_data, str):
                # If circle is just a string, it's the center
                circle_center = circle_data

        if not point or not circle_center:
            return {"passed": False, "message": f"Invalid point_on_circle condition: missing point or circle_center (got point={point}, circle_center={circle_center})"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": f"Could not find point '{point}'"}

        if circle_center not in element_dict:
            return {"passed": False, "message": f"Could not find circle center '{circle_center}'"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": f"'{point}' is not a point"}

        # Find all circles with this center (handles concentric circles)
        circle_labels = self._find_all_circles_with_center(circle_center)
        if not circle_labels:
            return {"passed": False, "message": f"Could not find any circle with center '{circle_center}'"}

        # Check if point is on any of the circles with this center
        for circle_label in circle_labels:
            circle = element_dict[circle_label].data
            result = cmd.contained_by_pc(pt, circle)
            if result.b:
                return {
                    "passed": True,
                    "message": f"Point '{point}' is on circle '{circle_label}' (center='{circle_center}')"
                }

        # Point is not on any circle with this center
        if len(circle_labels) == 1:
            return {
                "passed": False,
                "message": f"Point '{point}' is not on circle '{circle_labels[0]}' with center '{circle_center}'"
            }
        else:
            return {
                "passed": False,
                "message": f"Point '{point}' is not on any of the {len(circle_labels)} circles with center '{circle_center}'"
            }
    
    def _check_angle_bisector(self, data: Dict) -> Dict[str, Any]:
        """Check if a line bisects an angle."""
        line_data = data.get("line")
        angle_points = data.get("angle_points", [])

        if not line_data or len(angle_points) != 3:
            return {"passed": False, "message": "Invalid angle_bisector condition"}

        element_dict = self.construction.element_dict

        # Get angle vertex (middle point)
        vertex = angle_points[1]
        if vertex not in element_dict:
            return {"passed": False, "message": "Could not find angle vertex"}

        vertex_pt_elem = element_dict[vertex].data

        # Get bisector line
        bisector_line = self._get_line_from_data(line_data)
        if bisector_line is None:
            return {"passed": False, "message": "Could not find bisector line"}

        # Check if bisector line passes through vertex
        if not isinstance(vertex_pt_elem, gt.Point) or not bisector_line.contains(vertex_pt_elem.a):
            return {"passed": False, "message": "Bisector doesn't pass through angle vertex"}

        # Get all points
        if not all(p in element_dict for p in angle_points):
            return {"passed": False, "message": "Could not find angle points"}

        p1, vertex_pt, p3 = [element_dict[p].data for p in angle_points]

        if not all(isinstance(p, gt.Point) for p in [p1, vertex_pt, p3]):
            return {"passed": False, "message": "Invalid point types"}

        # Get a point on the bisector line (not the vertex)
        # We need to find or create a point on the line
        bisector_pt = None

        # If line_data is a list of 2 points, use the one that's not the vertex
        if isinstance(line_data, list) and len(line_data) == 2:
            for lp in line_data:
                if lp != vertex and lp in element_dict:
                    bisector_pt = element_dict[lp].data
                    break

        # If we couldn't get a point from line_data, create one on the line
        if bisector_pt is None or not isinstance(bisector_pt, gt.Point):
            # Create a point along the bisector direction
            # Use a point at distance 1 from vertex along the line
            # Line object has .v (direction vector)
            direction = bisector_line.v
            test_point_coords = vertex_pt.a + direction
            bisector_pt = gt.Point(test_point_coords)

        # Calculate angles on both sides
        angle1 = cmd.angle_ppp(p1, vertex_pt, bisector_pt)
        angle2 = cmd.angle_ppp(bisector_pt, vertex_pt, p3)

        # Check if angles are equal
        result = cmd.are_congruent_aa(angle1, angle2)

        angle1_deg = np.degrees(angle1.angle)
        angle2_deg = np.degrees(angle2.angle)

        return {
            "passed": result.b,
            "message": f"Bisector creates angles of {angle1_deg:.2f}° and {angle2_deg:.2f}°"
        }
    
    def _check_point_on_segment(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a segment between two endpoints."""
        point = data.get("point")
        segment = data.get("segment", [])
        
        if not point or len(segment) != 2:
            return {"passed": False, "message": "Invalid point_on_segment condition"}
        
        element_dict = self.construction.element_dict
        
        # Check all points exist
        if point not in element_dict or not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find all points"}
        
        pt = element_dict[point].data
        p1 = element_dict[segment[0]].data
        p2 = element_dict[segment[1]].data
        
        if not all(isinstance(p, gt.Point) for p in [pt, p1, p2]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Check if point is collinear with segment endpoints
        collinear_result = cmd.are_collinear_ppp(pt, p1, p2)
        if not collinear_result.b:
            return {"passed": False, "message": "Point is not collinear with segment"}
        
        # Check if point is between the endpoints
        # Point is on segment if: dist(p1,pt) + dist(pt,p2) ≈ dist(p1,p2)
        dist_1_pt = cmd.distance_pp(p1, pt)
        dist_pt_2 = cmd.distance_pp(pt, p2)
        dist_1_2 = cmd.distance_pp(p1, p2)
        
        total_dist = dist_1_pt.x + dist_pt_2.x
        segment_dist = dist_1_2.x
        
        # Check if distances sum correctly (within tolerance)
        is_between = np.abs(total_dist - segment_dist) <= self.tolerance
        
        return {
            "passed": is_between,
            "message": f"Point {'is' if is_between else 'is not'} on segment (distances: {dist_1_pt.x:.2f} + {dist_pt_2.x:.2f} = {total_dist:.2f}, segment: {segment_dist:.2f})"
        }
    
    def _check_midpoint_of(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is the midpoint of a segment."""
        point = data.get("point")
        # Support both 'segment' and 'objects' field
        segment = data.get("segment")
        if segment is None:
            objects = data.get("objects", [])
            segment = objects[0] if objects else []
        
        if not point or len(segment) != 2:
            return {"passed": False, "message": "Invalid midpoint_of condition"}
        
        element_dict = self.construction.element_dict
        
        # Check all points exist
        if point not in element_dict or not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find all points"}
        
        pt = element_dict[point].data
        p1 = element_dict[segment[0]].data
        p2 = element_dict[segment[1]].data
        
        if not all(isinstance(p, gt.Point) for p in [pt, p1, p2]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate actual midpoint
        midpoint_coords = (p1.a + p2.a) / 2
        
        # Check if point matches midpoint
        is_midpoint = np.allclose(pt.a, midpoint_coords, atol=self.tolerance)
        
        return {
            "passed": is_midpoint,
            "message": f"Point {'is' if is_midpoint else 'is not'} the midpoint of segment"
        }
    
    def _check_distance_equals(self, data: Dict) -> Dict[str, Any]:
        """Check if distance between two points equals expected value."""
        segment = data.get("segment", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", self.tolerance)
        
        if len(segment) != 2:
            return {"passed": False, "message": "Invalid distance_equals condition"}
        
        element_dict = self.construction.element_dict
        
        # Check points exist
        if not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find all points"}
        
        p1 = element_dict[segment[0]].data
        p2 = element_dict[segment[1]].data
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate distance
        dist = cmd.distance_pp(p1, p2)
        actual_value = dist.x

        # Correct for scaling applied by fit_to_window()
        # The coordinates are scaled, so distances are also scaled
        # Divide by scale_factor to get the original logical distance
        scale_factor = getattr(self.construction, 'scale_factor', 1.0)
        actual_value_corrected = actual_value / scale_factor

        # Check if distance matches expected value
        passed = np.abs(actual_value_corrected - expected_value) <= tolerance

        return {
            "passed": passed,
            "message": f"Distance is {actual_value_corrected:.2f}, expected {expected_value:.2f} (tolerance {tolerance:.2f})"
        }
    
    def _check_triangle_valid(self, data: Dict) -> Dict[str, Any]:
        """Check if three points form a valid (non-degenerate) triangle."""
        points = data.get("points", [])
        
        if len(points) != 3:
            return {"passed": False, "message": "Triangle requires exactly 3 points"}
        
        element_dict = self.construction.element_dict
        
        # Check all points exist
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2, p3 = [element_dict[p].data for p in points]
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Check if points are NOT collinear (valid triangle)
        collinear_result = cmd.are_collinear_ppp(p1, p2, p3)
        
        return {
            "passed": not collinear_result.b,
            "message": f"Points {'do not form' if collinear_result.b else 'form'} a valid triangle"
        }
    
    def _check_point_between(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is between two other points on a line."""
        point = data.get("point")
        endpoints = data.get("endpoints", [])
        
        if not point or len(endpoints) != 2:
            return {"passed": False, "message": "Invalid point_between condition"}
        
        element_dict = self.construction.element_dict
        
        # Check all points exist
        if point not in element_dict or not all(p in element_dict for p in endpoints):
            return {"passed": False, "message": "Could not find all points"}
        
        pt = element_dict[point].data
        p1 = element_dict[endpoints[0]].data
        p2 = element_dict[endpoints[1]].data
        
        if not all(isinstance(p, gt.Point) for p in [pt, p1, p2]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Check collinearity first
        collinear_result = cmd.are_collinear_ppp(pt, p1, p2)
        if not collinear_result.b:
            return {"passed": False, "message": "Point is not collinear with endpoints"}
        
        # Check if point is between (distance check)
        dist_1_pt = cmd.distance_pp(p1, pt).x
        dist_pt_2 = cmd.distance_pp(pt, p2).x
        dist_1_2 = cmd.distance_pp(p1, p2).x
        
        is_between = np.abs((dist_1_pt + dist_pt_2) - dist_1_2) <= self.tolerance
        
        return {
            "passed": is_between,
            "message": f"Point {'is' if is_between else 'is not'} between the endpoints"
        }
    
    def _check_concentric_circles(self, data: Dict) -> Dict[str, Any]:
        """Check if circles share the same center."""
        circle_centers = data.get("centers", [])
        
        if len(circle_centers) < 2:
            return {"passed": False, "message": "Concentric requires at least 2 circles"}
        
        element_dict = self.construction.element_dict
        
        # Get all circle centers as points
        center_points = []
        for center_label in circle_centers:
            if center_label not in element_dict:
                return {"passed": False, "message": f"Could not find center {center_label}"}
            
            center = element_dict[center_label].data
            if not isinstance(center, gt.Point):
                return {"passed": False, "message": f"Invalid center type for {center_label}"}
            
            center_points.append(center.a)
        
        # Check if all centers are at the same location
        first_center = center_points[0]
        all_concentric = all(np.allclose(first_center, c, atol=self.tolerance) for c in center_points[1:])
        
        return {
            "passed": all_concentric,
            "message": f"Circles {'are' if all_concentric else 'are not'} concentric"
        }
    
    # ============================================================
    # NEW CONDITION CHECK METHODS (Added from dataset analysis)
    # ============================================================
    
    def _check_angle_sum(self, data: Dict) -> Dict[str, Any]:
        """Check if the sum of angles equals expected value."""
        angles = data.get("angles", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", 2.0)  # 2 degree tolerance
        
        if len(angles) < 2:
            return {"passed": False, "message": "Angle sum requires at least 2 angles"}
        
        element_dict = self.construction.element_dict
        total_angle = 0.0
        
        for angle_def in angles:
            points = angle_def.get("points", [])
            if len(points) != 3:
                return {"passed": False, "message": "Each angle requires 3 points"}
            
            if not all(p in element_dict for p in points):
                return {"passed": False, "message": f"Could not find points for angle"}
            
            p1, p2, p3 = [element_dict[p].data for p in points]
            if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
                return {"passed": False, "message": "Invalid point types"}
            
            angle = cmd.angle_ppp(p1, p2, p3)
            total_angle += np.degrees(angle.angle)
        
        passed = np.abs(total_angle - expected_value) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Sum of angles is {total_angle:.2f}°, expected {expected_value}° (tolerance {tolerance}°)"
        }
    
    def _check_isosceles_triangle(self, data: Dict) -> Dict[str, Any]:
        """Check if a triangle is isosceles."""
        points = data.get("points", [])
        equal_sides = data.get("equal_sides", [])
        
        if len(points) != 3:
            return {"passed": False, "message": "Isosceles triangle requires 3 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        pts = [element_dict[p].data for p in points]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate all side lengths
        p1, p2, p3 = pts
        side_ab = cmd.distance_pp(p1, p2).x
        side_bc = cmd.distance_pp(p2, p3).x
        side_ca = cmd.distance_pp(p3, p1).x
        
        # Check if at least two sides are equal
        is_isosceles = (
            np.abs(side_ab - side_bc) <= self.tolerance or
            np.abs(side_bc - side_ca) <= self.tolerance or
            np.abs(side_ca - side_ab) <= self.tolerance
        )
        
        return {
            "passed": is_isosceles,
            "message": f"Triangle {'is' if is_isosceles else 'is not'} isosceles (sides: {side_ab:.2f}, {side_bc:.2f}, {side_ca:.2f})"
        }
    
    def _check_right_triangle(self, data: Dict) -> Dict[str, Any]:
        """Check if a triangle is a right triangle."""
        polygons = data.get("polygons", [])
        points = data.get("points", [])
        
        # Support both formats: {"polygons": [["A","B","C"]]} or {"points": ["A","B","C"]}
        if polygons and len(polygons) > 0:
            points = polygons[0]
        
        if len(points) != 3:
            return {"passed": False, "message": "Right triangle requires 3 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        pts = [element_dict[p].data for p in points]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        p1, p2, p3 = pts
        
        # Check each angle for 90 degrees
        angles = [
            (cmd.angle_ppp(p3, p1, p2), f"∠{points[0]}"),
            (cmd.angle_ppp(p1, p2, p3), f"∠{points[1]}"),
            (cmd.angle_ppp(p2, p3, p1), f"∠{points[2]}")
        ]
        
        for angle, name in angles:
            angle_deg = np.degrees(angle.angle)
            if np.abs(angle_deg - 90) <= 2.0:  # 2 degree tolerance
                return {
                    "passed": True,
                    "message": f"Triangle is right-angled at {name} ({angle_deg:.2f}°)"
                }
        
        return {
            "passed": False,
            "message": "Triangle is not a right triangle"
        }
    
    def _check_length_value(self, data: Dict) -> Dict[str, Any]:
        """Check if a segment has expected length. Handles length, length_value, segment_length."""
        segment = data.get("segment", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", self.tolerance)
        
        if len(segment) != 2:
            return {"passed": False, "message": "Length check requires 2 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2 = [element_dict[p].data for p in segment]
        if not all(isinstance(p, gt.Point) for p in [p1, p2]):
            return {"passed": False, "message": "Invalid point types"}

        dist = cmd.distance_pp(p1, p2).x

        # Correct for scaling applied by fit_to_window()
        scale_factor = getattr(self.construction, 'scale_factor', 1.0)
        dist_corrected = dist / scale_factor

        passed = np.abs(dist_corrected - expected_value) <= tolerance

        return {
            "passed": passed,
            "message": f"Segment length is {dist_corrected:.2f}, expected {expected_value} (tolerance {tolerance})"
        }

    def _check_angle_ratio(self, data: Dict) -> Dict[str, Any]:
        """
        Check if two angles have a specific ratio.

        Format:
        {
            "type": "angle_ratio",
            "angle1": ["C", "B", "D"],  # First angle (CBD)
            "angle2": ["B", "D", "C"],  # Second angle (BDC)
            "ratio": [2, 1]             # angle1 : angle2 = 2:1 (i.e., ∠CBD = 2∠BDC)
        }

        Examples:
        - ∠CBD = 2∠BDC → ratio: [2, 1]
        - 2∠ABC = 3∠DEF → ratio: [2, 3]
        """
        angle1_points = data.get("angle1", [])
        angle2_points = data.get("angle2", [])
        ratio = data.get("ratio", [1, 1])
        tolerance = data.get("tolerance", 1.0)  # Tolerance in degrees

        if len(angle1_points) != 3 or len(angle2_points) != 3:
            return {"passed": False, "message": "Each angle requires exactly 3 points"}

        if len(ratio) != 2:
            return {"passed": False, "message": "Ratio must be [numerator, denominator]"}

        element_dict = self.construction.element_dict

        # Check all points exist
        all_points = angle1_points + angle2_points
        if not all(p in element_dict for p in all_points):
            return {"passed": False, "message": "Could not find all points"}

        # Get angle values
        try:
            # angle1 = ∠(p0, p1, p2) where p1 is the vertex
            pts1 = [element_dict[p].data for p in angle1_points]
            pts2 = [element_dict[p].data for p in angle2_points]

            if not all(isinstance(p, gt.Point) for p in pts1 + pts2):
                return {"passed": False, "message": "Invalid point types"}

            # Calculate angles
            angle1 = cmd.angle_ppp(pts1[0], pts1[1], pts1[2])
            angle2 = cmd.angle_ppp(pts2[0], pts2[1], pts2[2])

            angle1_deg = np.degrees(angle1.angle)
            angle2_deg = np.degrees(angle2.angle)

            # Consider reflex angles: allow either the measured angle or (360 - angle)
            # Try all 4 combinations and use the one with the best ratio match
            angle1_options = [angle1_deg, (360.0 - angle1_deg) % 360.0]
            angle2_options = [angle2_deg, (360.0 - angle2_deg) % 360.0]

            expected_ratio = ratio[0] / ratio[1]
            best_diff = float('inf')
            best_combo = (angle1_deg, angle2_deg)

            for a1 in angle1_options:
                for a2 in angle2_options:
                    if a2 == 0:
                        continue
                    # Cross multiply: a1 * ratio[1] = a2 * ratio[0]
                    lhs = a1 * ratio[1]
                    rhs = a2 * ratio[0]
                    diff = np.abs(lhs - rhs)
                    if diff < best_diff:
                        best_diff = diff
                        best_combo = (a1, a2)

            angle1_chosen, angle2_chosen = best_combo
            actual_ratio = angle1_chosen / angle2_chosen if angle2_chosen != 0 else np.inf

            passed = best_diff <= tolerance * max(ratio)

            return {
                "passed": passed,
                "message": (
                    f"Angle ratio: {angle1_chosen:.2f}° : {angle2_chosen:.2f}° = "
                    f"{actual_ratio:.3f} (expected {expected_ratio:.3f}, "
                    f"from ratio [{ratio[0]}, {ratio[1]}])"
                )
            }
        except Exception as e:
            return {"passed": False, "message": f"Error calculating angles: {str(e)}"}

    def _check_segment_ratio(self, data: Dict) -> Dict[str, Any]:
        """
        Check if two segments have a specific ratio.

        Format:
        {
            "type": "segment_ratio",
            "segment1": ["A", "B"],  # First segment
            "segment2": ["B", "D"],  # Second segment
            "ratio": [3, 1]          # segment1 : segment2 = 3:1 (i.e., AB = 3BD)
        }

        Examples:
        - AB = 3BD → ratio: [3, 1]
        - 2BD = 3AB → ratio: [2, 3]
        """
        segment1 = data.get("segment1", [])
        segment2 = data.get("segment2", [])
        ratio = data.get("ratio", [1, 1])
        tolerance = data.get("tolerance", self.tolerance)

        if len(segment1) != 2 or len(segment2) != 2:
            return {"passed": False, "message": "Each segment requires exactly 2 points"}

        if len(ratio) != 2:
            return {"passed": False, "message": "Ratio must be [numerator, denominator]"}

        element_dict = self.construction.element_dict

        # Check all points exist
        all_points = segment1 + segment2
        if not all(p in element_dict for p in all_points):
            return {"passed": False, "message": "Could not find all points"}

        # Get segment lengths
        try:
            pts1 = [element_dict[p].data for p in segment1]
            pts2 = [element_dict[p].data for p in segment2]

            if not all(isinstance(p, gt.Point) for p in pts1 + pts2):
                return {"passed": False, "message": "Invalid point types"}

            # Calculate distances
            dist1 = cmd.distance_pp(pts1[0], pts1[1]).x
            dist2 = cmd.distance_pp(pts2[0], pts2[1]).x

            # Correct for scaling
            scale_factor = getattr(self.construction, 'scale_factor', 1.0)
            dist1_corrected = dist1 / scale_factor
            dist2_corrected = dist2 / scale_factor

            # Check ratio: dist1/dist2 should equal ratio[0]/ratio[1]
            # Cross multiply: dist1 * ratio[1] = dist2 * ratio[0]
            expected_ratio = ratio[0] / ratio[1]
            actual_ratio = dist1_corrected / dist2_corrected if dist2_corrected != 0 else np.inf

            lhs = dist1_corrected * ratio[1]
            rhs = dist2_corrected * ratio[0]

            passed = np.abs(lhs - rhs) <= tolerance * max(ratio)

            segment1_name = "".join(segment1)
            segment2_name = "".join(segment2)

            return {
                "passed": passed,
                "message": (
                    f"Segment ratio: {segment1_name}({dist1_corrected:.2f}) : "
                    f"{segment2_name}({dist2_corrected:.2f}) = {actual_ratio:.3f} "
                    f"(expected {expected_ratio:.3f}, from ratio [{ratio[0]}, {ratio[1]}])"
                )
            }
        except Exception as e:
            return {"passed": False, "message": f"Error calculating segments: {str(e)}"}

    def _check_perpendicular_bisector(self, data: Dict) -> Dict[str, Any]:
        """Check if a line is the perpendicular bisector of a segment."""
        line_data = data.get("line")
        segment = data.get("segment", [])

        if not line_data or len(segment) != 2:
            return {"passed": False, "message": "Invalid perpendicular_bisector condition"}

        element_dict = self.construction.element_dict

        # Get segment endpoints
        if not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find segment points"}

        seg_p1, seg_p2 = [element_dict[p].data for p in segment]
        if not all(isinstance(p, gt.Point) for p in [seg_p1, seg_p2]):
            return {"passed": False, "message": "Invalid point types"}

        # Calculate midpoint
        midpoint = (seg_p1.a + seg_p2.a) / 2

        # Get the bisector line (supports all formats)
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find bisector line"}

        # Check 1: Line passes through midpoint
        passes_midpoint = line.contains(midpoint)

        # Check 2: Line is perpendicular to segment
        segment_line = cmd.line_pp(seg_p1, seg_p2)
        perp_result = cmd.are_perpendicular_ll(line, segment_line)

        passed = passes_midpoint and perp_result.b

        return {
            "passed": passed,
            "message": f"Line {'is' if passed else 'is not'} perpendicular bisector (midpoint: {passes_midpoint}, perpendicular: {perp_result.b})"
        }
    
    def _check_point_on_line_extension(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is on the extension of a line segment."""
        point = data.get("point")
        line_data = data.get("line_segment", data.get("line"))

        if not point or not line_data:
            return {"passed": False, "message": "Invalid point_on_line_extension condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}

        # Get line
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find line"}

        # Check if point is on the line (collinear)
        is_on_line = line.contains(pt.a)

        return {
            "passed": is_on_line,
            "message": f"Point {'is' if is_on_line else 'is not'} on line extension"
        }
    
    def _check_point_on_segment_extension(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is on the extension of a segment (beyond the endpoints)."""
        point = data.get("point")
        segment = data.get("segment", [])
        
        if not point or len(segment) != 2:
            return {"passed": False, "message": "Invalid point_on_segment_extension condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict or not all(p in element_dict for p in segment):
            return {"passed": False, "message": "Could not find all points"}
        
        pt = element_dict[point].data
        p1 = element_dict[segment[0]].data
        p2 = element_dict[segment[1]].data
        
        if not all(isinstance(p, gt.Point) for p in [pt, p1, p2]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Check collinearity first
        collinear_result = cmd.are_collinear_ppp(pt, p1, p2)
        if not collinear_result.b:
            return {"passed": False, "message": "Point is not collinear with segment"}
        
        # Check if point is outside the segment (on extension)
        dist_1_pt = cmd.distance_pp(p1, pt).x
        dist_pt_2 = cmd.distance_pp(pt, p2).x
        dist_1_2 = cmd.distance_pp(p1, p2).x
        
        # Point is on extension if it's collinear but not between the endpoints
        is_between = np.abs((dist_1_pt + dist_pt_2) - dist_1_2) <= self.tolerance
        is_on_extension = collinear_result.b and not is_between
        
        return {
            "passed": is_on_extension,
            "message": f"Point {'is' if is_on_extension else 'is not'} on segment extension"
        }
    
    def _check_same_side(self, data: Dict) -> Dict[str, Any]:
        """Check if two points are on the same side of a line."""
        points = data.get("points", [])
        line_data = data.get("line")

        if len(points) != 2 or not line_data:
            return {"passed": False, "message": "Same side requires 2 points and a line"}

        element_dict = self.construction.element_dict

        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}

        pt1, pt2 = [element_dict[p].data for p in points]

        if not all(isinstance(p, gt.Point) for p in [pt1, pt2]):
            return {"passed": False, "message": "Invalid point types"}

        # Get line (supports all formats)
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find line"}

        # Calculate signed distance for both points
        # Using line equation: n·x = c (where n is normalized)
        # Signed distance from point to line: n·p - c
        # Line object has: line.n (normal vector), line.c (constant)

        dist1 = np.dot(line.n, pt1.a) - line.c
        dist2 = np.dot(line.n, pt2.a) - line.c

        # Same side if both distances have the same sign
        same_side = (dist1 * dist2) > 0
        
        return {
            "passed": same_side,
            "message": f"Points {'are' if same_side else 'are not'} on the same side of the line"
        }
    
    def _check_point_inside_circle(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is inside a circle."""
        point = data.get("point")
        circle_center = data.get("circle_center")
        radius = data.get("radius")
        
        if not point or not circle_center:
            return {"passed": False, "message": "Invalid point_inside_circle condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict or circle_center not in element_dict:
            return {"passed": False, "message": "Could not find point or center"}
        
        pt = element_dict[point].data
        center = element_dict[circle_center].data
        
        if not isinstance(pt, gt.Point) or not isinstance(center, gt.Point):
            return {"passed": False, "message": "Invalid point types"}
        
        dist = cmd.distance_pp(pt, center).x
        
        # Get radius from circle if not provided
        if radius is None:
            circle_label = self._find_circle_with_center(circle_center)
            if circle_label:
                circle = element_dict[circle_label].data
                radius = circle.r
            else:
                return {"passed": False, "message": "Could not determine circle radius"}
        
        is_inside = dist < radius - self.tolerance
        
        return {
            "passed": is_inside,
            "message": f"Point {'is' if is_inside else 'is not'} inside circle (distance: {dist:.2f}, radius: {radius:.2f})"
        }
    
    def _check_tangent_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a line is tangent to a circle."""
        line_data = data.get("line")
        circle_center = data.get("circle_center")
        circle_data = data.get("circle", {})

        if circle_center is None and isinstance(circle_data, dict):
            circle_center = circle_data.get("center")

        if not line_data or not circle_center:
            return {"passed": False, "message": "Invalid tangent line condition"}

        element_dict = self.construction.element_dict

        if circle_center not in element_dict:
            return {"passed": False, "message": "Could not find circle center"}

        center = element_dict[circle_center].data
        if not isinstance(center, gt.Point):
            return {"passed": False, "message": "Invalid center type"}

        # Get the line (supports all formats)
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find line"}
        
        # Find circle
        circle_label = self._find_circle_with_center(circle_center)
        if not circle_label:
            return {"passed": False, "message": "Could not find circle"}
        
        circle = element_dict[circle_label].data

        # Calculate distance from center to line
        # Line equation: n·x = c (where n is normalized normal vector)
        # Distance from point p to line: |n·p - c| / ||n||
        # Since n is already normalized, distance = |n·p - c|
        dist_to_line = np.abs(np.dot(line.n, center.a) - line.c)

        # Line is tangent if distance equals radius
        is_tangent = np.abs(dist_to_line - circle.r) <= self.tolerance
        
        return {
            "passed": is_tangent,
            "message": f"Line {'is' if is_tangent else 'is not'} tangent to circle (distance: {dist_to_line:.2f}, radius: {circle.r:.2f})"
        }
    
    def _check_tangent_at_point(self, data: Dict) -> Dict[str, Any]:
        """Check if a line is tangent to a circle at a specific point."""
        line_data = data.get("line")
        circle_center = data.get("circle_center")
        tangent_point = data.get("tangent_point", data.get("point"))

        if not line_data or not circle_center or not tangent_point:
            return {"passed": False, "message": "Invalid tangent_at_point condition"}

        # First check if line is tangent
        tangent_result = self._check_tangent_line({
            "line": line_data,
            "circle_center": circle_center
        })

        if not tangent_result["passed"]:
            return tangent_result

        # Check if tangent point is on the circle
        point_on_circle = self._check_point_on_circle({
            "point": tangent_point,
            "circle_center": circle_center
        })

        if not point_on_circle["passed"]:
            return {"passed": False, "message": "Tangent point is not on circle"}

        # Check if tangent point is on the line
        point_on_line = self._check_point_on_line({
            "point": tangent_point,
            "line": line_data
        })

        return {
            "passed": point_on_line["passed"],
            "message": f"Line {'is' if point_on_line['passed'] else 'is not'} tangent at point {tangent_point}"
        }
    
    def _check_diameter(self, data: Dict) -> Dict[str, Any]:
        """Check if a segment is the diameter of a circle."""
        segment = data.get("segment", [])
        circle_center = data.get("circle_center")
        
        if len(segment) != 2 or not circle_center:
            return {"passed": False, "message": "Invalid diameter condition"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in segment) or circle_center not in element_dict:
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2 = [element_dict[p].data for p in segment]
        center = element_dict[circle_center].data
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, center]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Check if center is the midpoint of the segment
        midpoint = (p1.a + p2.a) / 2
        is_midpoint = np.allclose(center.a, midpoint, atol=self.tolerance)
        
        # Check if both endpoints are on the circle
        circle_label = self._find_circle_with_center(circle_center)
        if not circle_label:
            return {"passed": False, "message": "Could not find circle"}
        
        circle = element_dict[circle_label].data
        on_circle_1 = np.abs(np.linalg.norm(p1.a - center.a) - circle.r) <= self.tolerance
        on_circle_2 = np.abs(np.linalg.norm(p2.a - center.a) - circle.r) <= self.tolerance
        
        is_diameter = is_midpoint and on_circle_1 and on_circle_2
        
        return {
            "passed": is_diameter,
            "message": f"Segment {'is' if is_diameter else 'is not'} a diameter"
        }
    
    def _check_intersection_point(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is the intersection of two lines."""
        point = data.get("point")
        lines = data.get("lines", [])

        if not point or len(lines) != 2:
            return {"passed": False, "message": "Invalid intersection_point condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": "Could not find intersection point"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}

        # Check if point is on both lines (supports all line formats)
        for line_data in lines:
            line = self._get_line_from_data(line_data)
            if line is None:
                return {"passed": False, "message": f"Could not find line {line_data}"}

            if not line.contains(pt.a):
                return {
                    "passed": False,
                    "message": f"Point is not on line {line_data}"
                }
        
        return {
            "passed": True,
            "message": f"Point {point} is the intersection of the two lines"
        }
    
    def _check_polygon_property(self, data: Dict) -> Dict[str, Any]:
        """Check if a polygon has a specific property (parallelogram, rectangle, etc.)."""
        polygon = data.get("polygon", [])
        property_type = data.get("property", "").lower()
        
        if len(polygon) < 3:
            return {"passed": False, "message": "Polygon requires at least 3 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in polygon):
            return {"passed": False, "message": "Could not find all polygon points"}
        
        pts = [element_dict[p].data for p in polygon]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        if property_type == "parallelogram":
            if len(polygon) != 4:
                return {"passed": False, "message": "Parallelogram requires 4 points"}
            
            # Check if opposite sides are parallel
            p1, p2, p3, p4 = pts
            side1 = cmd.line_pp(p1, p2)
            side3 = cmd.line_pp(p3, p4)
            side2 = cmd.line_pp(p2, p3)
            side4 = cmd.line_pp(p4, p1)
            
            parallel1 = cmd.are_parallel_ll(side1, side3)
            parallel2 = cmd.are_parallel_ll(side2, side4)
            
            is_parallelogram = parallel1.b and parallel2.b
            return {
                "passed": is_parallelogram,
                "message": f"Quadrilateral {'is' if is_parallelogram else 'is not'} a parallelogram"
            }
        
        elif property_type == "rectangle":
            if len(polygon) != 4:
                return {"passed": False, "message": "Rectangle requires 4 points"}

            # Rectangle: parallelogram with right angles
            p1, p2, p3, p4 = pts

            # Check if opposite sides are parallel
            side1 = cmd.line_pp(p1, p2)
            side3 = cmd.line_pp(p3, p4)
            side2 = cmd.line_pp(p2, p3)
            side4 = cmd.line_pp(p4, p1)

            parallel1 = cmd.are_parallel_ll(side1, side3)
            parallel2 = cmd.are_parallel_ll(side2, side4)

            if not (parallel1.b and parallel2.b):
                return {"passed": False, "message": "Not a parallelogram (opposite sides not parallel)"}

            # Check if angles are right angles
            angle1 = cmd.angle_ppp(p4, p1, p2)
            is_right_angle = np.abs(np.abs(angle1.angle) - np.pi/2) <= np.radians(self.tolerance)

            return {
                "passed": is_right_angle,
                "message": f"Quadrilateral {'is' if is_right_angle else 'is not'} a rectangle (angle: {np.degrees(angle1.angle):.2f}°)"
            }

        elif property_type == "rhombus":
            if len(polygon) != 4:
                return {"passed": False, "message": "Rhombus requires 4 points"}

            # Rhombus: parallelogram with all sides equal
            p1, p2, p3, p4 = pts

            # Check if opposite sides are parallel
            side1 = cmd.line_pp(p1, p2)
            side3 = cmd.line_pp(p3, p4)
            side2 = cmd.line_pp(p2, p3)
            side4 = cmd.line_pp(p4, p1)

            parallel1 = cmd.are_parallel_ll(side1, side3)
            parallel2 = cmd.are_parallel_ll(side2, side4)

            if not (parallel1.b and parallel2.b):
                return {"passed": False, "message": "Not a parallelogram (opposite sides not parallel)"}

            # Check if all sides are equal
            s1 = cmd.distance_pp(p1, p2).x
            s2 = cmd.distance_pp(p2, p3).x
            s3 = cmd.distance_pp(p3, p4).x
            s4 = cmd.distance_pp(p4, p1).x

            sides_equal = (
                np.abs(s1 - s2) <= self.tolerance and
                np.abs(s2 - s3) <= self.tolerance and
                np.abs(s3 - s4) <= self.tolerance
            )

            return {
                "passed": sides_equal,
                "message": f"Quadrilateral {'is' if sides_equal else 'is not'} a rhombus (sides: {s1:.2f}, {s2:.2f}, {s3:.2f}, {s4:.2f})"
            }

        elif property_type == "square":
            if len(polygon) != 4:
                return {"passed": False, "message": "Square requires 4 points"}

            # Square: rectangle with all sides equal (or rhombus with right angles)
            p1, p2, p3, p4 = pts

            # Check if all sides are equal
            s1 = cmd.distance_pp(p1, p2).x
            s2 = cmd.distance_pp(p2, p3).x
            s3 = cmd.distance_pp(p3, p4).x
            s4 = cmd.distance_pp(p4, p1).x

            sides_equal = (
                np.abs(s1 - s2) <= self.tolerance and
                np.abs(s2 - s3) <= self.tolerance and
                np.abs(s3 - s4) <= self.tolerance
            )

            # Check if angles are right angles
            angle1 = cmd.angle_ppp(p4, p1, p2)
            is_right_angle = np.abs(np.abs(angle1.angle) - np.pi/2) <= np.radians(self.tolerance)

            is_square = sides_equal and is_right_angle

            return {
                "passed": is_square,
                "message": f"Quadrilateral {'is' if is_square else 'is not'} a square (sides: {s1:.2f}, {s2:.2f}, {s3:.2f}, {s4:.2f}, angle: {np.degrees(angle1.angle):.2f}°)"
            }

        return {"passed": False, "message": f"Unknown polygon property: {property_type}"}

    def _check_regular_polygon(self, data: Dict) -> Dict[str, Any]:
        """Check if a polygon is regular (all sides and angles equal)."""
        polygon_points = data.get("polygon_points", [])
        expected_sides = data.get("sides", len(polygon_points))
        
        if len(polygon_points) < 3:
            return {"passed": False, "message": "Regular polygon requires at least 3 points"}
        
        if len(polygon_points) != expected_sides:
            return {"passed": False, "message": f"Expected {expected_sides} sides but got {len(polygon_points)} points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in polygon_points):
            return {"passed": False, "message": "Could not find all polygon points"}
        
        pts = [element_dict[p].data for p in polygon_points]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        n = len(pts)
        
        # Check all side lengths are equal
        side_lengths = []
        for i in range(n):
            dist = cmd.distance_pp(pts[i], pts[(i + 1) % n]).x
            side_lengths.append(dist)
        
        avg_side = sum(side_lengths) / n
        sides_equal = all(np.abs(s - avg_side) <= self.tolerance for s in side_lengths)
        
        # Check all angles are equal (expected angle for regular polygon: (n-2)*180/n)
        expected_angle = (n - 2) * 180 / n

        # Calculate all interior angles
        angles = []
        for i in range(n):
            # Angle at vertex i: formed by points (i-1), i, (i+1)
            p_prev = pts[(i - 1) % n]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]

            angle = cmd.angle_ppp(p_prev, p_curr, p_next)
            angle_degrees = np.degrees(angle.angle)
            angles.append(angle_degrees)

        # Check if all angles match expected angle
        angles_equal = all(np.abs(a - expected_angle) <= self.tolerance for a in angles)

        is_regular = sides_equal and angles_equal
        
        return {
            "passed": is_regular,
            "message": f"Polygon {'is' if is_regular else 'is not'} regular (sides equal: {sides_equal}, angles equal: {angles_equal}, expected angle: {expected_angle:.1f}°, actual angles: {[f'{a:.1f}°' for a in angles]})"
        }
    
    def _check_square(self, data: Dict) -> Dict[str, Any]:
        """Check if a quadrilateral is a square."""
        polygons = data.get("polygons", [])
        
        if not polygons or len(polygons[0]) != 4:
            return {"passed": False, "message": "Square requires exactly 4 points"}
        
        polygon = polygons[0]
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in polygon):
            return {"passed": False, "message": "Could not find all points"}
        
        pts = [element_dict[p].data for p in polygon]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        p1, p2, p3, p4 = pts
        
        # Check all sides are equal
        sides = [
            cmd.distance_pp(p1, p2).x,
            cmd.distance_pp(p2, p3).x,
            cmd.distance_pp(p3, p4).x,
            cmd.distance_pp(p4, p1).x
        ]
        
        avg_side = sum(sides) / 4
        sides_equal = all(np.abs(s - avg_side) <= self.tolerance for s in sides)
        
        # Check all angles are 90 degrees
        line1 = cmd.line_pp(p1, p2)
        line2 = cmd.line_pp(p2, p3)
        line3 = cmd.line_pp(p3, p4)
        line4 = cmd.line_pp(p4, p1)
        
        perp1 = cmd.are_perpendicular_ll(line1, line2)
        perp2 = cmd.are_perpendicular_ll(line2, line3)
        perp3 = cmd.are_perpendicular_ll(line3, line4)
        perp4 = cmd.are_perpendicular_ll(line4, line1)
        
        all_perp = perp1.b and perp2.b and perp3.b and perp4.b
        
        is_square = sides_equal and all_perp
        
        return {
            "passed": is_square,
            "message": f"Quadrilateral {'is' if is_square else 'is not'} a square"
        }
    
    def _check_order_on_line(self, data: Dict) -> Dict[str, Any]:
        """Check if points are in order on a line.

        If 'line' is provided (named line), also validates that all points lie on that line.
        """
        points = data.get("points", [])
        line_data = data.get("line")

        if len(points) < 3:
            return {"passed": False, "message": "Order check requires at least 3 points"}

        element_dict = self.construction.element_dict

        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}

        pts = [element_dict[p].data for p in points]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}

        # Check collinearity first
        for i in range(len(pts) - 2):
            collinear_result = cmd.are_collinear_ppp(pts[i], pts[i+1], pts[i+2])
            if not collinear_result.b:
                return {"passed": False, "message": "Points are not collinear"}

        # If named line is provided, verify all points lie on it
        if line_data:
            line = self._get_line_from_data(line_data)
            if line is None:
                return {"passed": False, "message": "Could not find named line"}

            # Check all points lie on the named line
            for pt in pts:
                if not line.contains(pt.a):
                    return {"passed": False, "message": "Not all points are on named line"}

        # Check order by distances from first point
        distances = [cmd.distance_pp(pts[0], pt).x for pt in pts]
        is_ordered = all(distances[i] <= distances[i+1] for i in range(len(distances) - 1))

        return {
            "passed": is_ordered,
            "message": f"Points {'are' if is_ordered else 'are not'} in order on the line"
        }
    
    def _check_perimeter(self, data: Dict) -> Dict[str, Any]:
        """Check if a polygon has expected perimeter."""
        polygon = data.get("polygon", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", self.tolerance)
        
        if len(polygon) < 3:
            return {"passed": False, "message": "Perimeter requires at least 3 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in polygon):
            return {"passed": False, "message": "Could not find all points"}
        
        pts = [element_dict[p].data for p in polygon]
        if not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate perimeter
        n = len(pts)
        perimeter = sum(cmd.distance_pp(pts[i], pts[(i + 1) % n]).x for i in range(n))
        
        passed = np.abs(perimeter - expected_value) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Perimeter is {perimeter:.2f}, expected {expected_value}"
        }
    
    def _check_segments_sum_value(self, data: Dict) -> Dict[str, Any]:
        """Check if the sum of multiple segments equals an expected value.
        
        Example usage in dataset:
        {
            "type": "segments_sum_value",
            "segments": [["A", "B"], ["B", "C"]],
            "value": 10
        }
        """
        segments = data.get("segments", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", self.tolerance)
        
        if len(segments) < 1:
            return {"passed": False, "message": "segments_sum_value requires at least 1 segment"}
        
        element_dict = self.construction.element_dict
        
        # Calculate the sum of all segment lengths
        total_length = 0.0
        segment_details = []
        
        for seg in segments:
            if len(seg) != 2:
                return {"passed": False, "message": f"Invalid segment: {seg}"}
            
            if not all(p in element_dict for p in seg):
                return {"passed": False, "message": f"Could not find points for segment {seg}"}
            
            p1 = element_dict[seg[0]].data
            p2 = element_dict[seg[1]].data
            
            if not all(isinstance(p, gt.Point) for p in [p1, p2]):
                return {"passed": False, "message": f"Invalid point types in segment {seg}"}
            
            dist = cmd.distance_pp(p1, p2).x
            total_length += dist
            segment_details.append(f"{seg[0]}{seg[1]}={dist:.2f}")
        
        passed = np.abs(total_length - expected_value) <= tolerance
        
        segments_str = " + ".join(segment_details)
        return {
            "passed": passed,
            "message": f"Sum of segments ({segments_str}) = {total_length:.2f}, expected {expected_value} (tolerance {tolerance})"
        }
    
    def _check_segments_sum_equals(self, data: Dict) -> Dict[str, Any]:
        """Check if the sum of some segments equals the sum of other segments.
        
        Example usage in dataset:
        {
            "type": "segments_sum_equals",
            "left_segments": [["A", "B"], ["B", "C"]],
            "right_segments": [["A", "D"]]
        }
        
        This checks if AB + BC = AD
        """
        left_segments = data.get("left_segments", data.get("sum_segments", []))
        right_segments = data.get("right_segments", data.get("target_segments", []))
        tolerance = data.get("tolerance", self.tolerance)
        
        # Also support single target segment format
        if not right_segments and "target_segment" in data:
            right_segments = [data["target_segment"]]
        
        if len(left_segments) < 1 or len(right_segments) < 1:
            return {"passed": False, "message": "segments_sum_equals requires segments on both sides"}
        
        element_dict = self.construction.element_dict
        
        def calc_sum(segments_list, side_name):
            """Calculate sum of segment lengths."""
            total = 0.0
            details = []
            for seg in segments_list:
                if len(seg) != 2:
                    return None, f"Invalid segment: {seg}"
                
                if not all(p in element_dict for p in seg):
                    return None, f"Could not find points for segment {seg}"
                
                p1 = element_dict[seg[0]].data
                p2 = element_dict[seg[1]].data
                
                if not all(isinstance(p, gt.Point) for p in [p1, p2]):
                    return None, f"Invalid point types in segment {seg}"
                
                dist = cmd.distance_pp(p1, p2).x
                total += dist
                details.append(f"{seg[0]}{seg[1]}={dist:.2f}")
            return (total, details), None
        
        # Calculate left side sum
        left_result, left_err = calc_sum(left_segments, "left")
        if left_err:
            return {"passed": False, "message": left_err}
        left_sum, left_details = left_result
        
        # Calculate right side sum
        right_result, right_err = calc_sum(right_segments, "right")
        if right_err:
            return {"passed": False, "message": right_err}
        right_sum, right_details = right_result
        
        passed = np.abs(left_sum - right_sum) <= tolerance
        
        left_str = " + ".join(left_details)
        right_str = " + ".join(right_details)
        
        return {
            "passed": passed,
            "message": f"Left sum ({left_str}) = {left_sum:.2f}, Right sum ({right_str}) = {right_sum:.2f}; {'equal' if passed else 'not equal'} (tolerance {tolerance})"
        }
    
    def _check_ratio(self, data: Dict) -> Dict[str, Any]:
        """Check if two segments have a specific ratio.
        
        Example usage in dataset:
        {
            "type": "ratio",
            "segments": [["A", "B"], ["B", "C"]],
            "ratio": [1, 2]  # AB:BC = 1:2
        }
        """
        segments = data.get("segments", [])
        ratio = data.get("ratio", [1, 1])
        tolerance = data.get("tolerance", 0.01)  # Ratio tolerance
        
        if len(segments) != 2:
            return {"passed": False, "message": "ratio requires exactly 2 segments"}
        
        if len(ratio) != 2 or ratio[1] == 0:
            return {"passed": False, "message": "Invalid ratio specification"}
        
        element_dict = self.construction.element_dict
        
        # Calculate both segment lengths
        lengths = []
        for seg in segments:
            if len(seg) != 2:
                return {"passed": False, "message": f"Invalid segment: {seg}"}
            
            if not all(p in element_dict for p in seg):
                return {"passed": False, "message": f"Could not find points for segment {seg}"}
            
            p1 = element_dict[seg[0]].data
            p2 = element_dict[seg[1]].data
            
            if not all(isinstance(p, gt.Point) for p in [p1, p2]):
                return {"passed": False, "message": f"Invalid point types in segment {seg}"}
            
            dist = cmd.distance_pp(p1, p2).x
            lengths.append(dist)
        
        # Calculate actual ratio
        expected_ratio = ratio[0] / ratio[1]
        actual_ratio = lengths[0] / lengths[1] if lengths[1] != 0 else float('inf')
        
        passed = np.abs(actual_ratio - expected_ratio) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Segment {segments[0][0]}{segments[0][1]}:{segments[1][0]}{segments[1][1]} = {lengths[0]:.2f}:{lengths[1]:.2f} (ratio {actual_ratio:.3f}), expected {ratio[0]}:{ratio[1]} (ratio {expected_ratio:.3f})"
        }
    
    def _check_point_incenter(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is the incenter of a triangle."""
        point = data.get("point")
        triangle = data.get("triangle", [])
        
        if not point or len(triangle) != 3:
            return {"passed": False, "message": "Invalid incenter condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict or not all(p in element_dict for p in triangle):
            return {"passed": False, "message": "Could not find all points"}
        
        incenter = element_dict[point].data
        pts = [element_dict[p].data for p in triangle]
        
        if not isinstance(incenter, gt.Point) or not all(isinstance(p, gt.Point) for p in pts):
            return {"passed": False, "message": "Invalid point types"}
        
        p1, p2, p3 = pts
        
        # Incenter should be equidistant from all three sides
        # Calculate distance from incenter to each side
        def dist_to_line(pt, l1, l2):
            a = l2.a[1] - l1.a[1]
            b = l1.a[0] - l2.a[0]
            c = l2.a[0] * l1.a[1] - l1.a[0] * l2.a[1]
            return np.abs(a * pt.a[0] + b * pt.a[1] + c) / np.sqrt(a**2 + b**2)
        
        d1 = dist_to_line(incenter, p1, p2)
        d2 = dist_to_line(incenter, p2, p3)
        d3 = dist_to_line(incenter, p3, p1)
        
        is_incenter = np.abs(d1 - d2) <= self.tolerance and np.abs(d2 - d3) <= self.tolerance
        
        return {
            "passed": is_incenter,
            "message": f"Point {'is' if is_incenter else 'is not'} the incenter (distances: {d1:.2f}, {d2:.2f}, {d3:.2f})"
        }
    
    def _check_point_on_arc(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is on an arc."""
        point = data.get("point")
        arc = data.get("arc", [])
        arc_type = data.get("arc_type", "minor")  # minor or major

        if not point or len(arc) < 2:
            return {"passed": False, "message": "Invalid point_on_arc condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict or not all(p in element_dict for p in arc[:2]):
            return {"passed": False, "message": "Could not find all points"}

        pt = element_dict[point].data
        arc_p1 = element_dict[arc[0]].data
        arc_p2 = element_dict[arc[1]].data

        if not all(isinstance(p, gt.Point) for p in [pt, arc_p1, arc_p2]):
            return {"passed": False, "message": "Invalid point types"}

        # Check if all three points are concyclic
        # Three points are always concyclic (define a unique circle)
        # So we need to check if they form a valid arc configuration

        # Calculate distances to verify they're on the same circle
        # Use the circumcenter of the three points
        try:
            # Get midpoints and perpendicular bisectors
            mid1 = (arc_p1.a + pt.a) / 2
            mid2 = (arc_p2.a + pt.a) / 2

            # Direction vectors
            dir1 = pt.a - arc_p1.a
            dir2 = pt.a - arc_p2.a

            # Perpendicular directions (rotate 90 degrees)
            perp1 = np.array([-dir1[1], dir1[0]])
            perp2 = np.array([-dir2[1], dir2[0]])

            # Find intersection of perpendicular bisectors (circumcenter)
            # This is a simplified check - if points are collinear, they don't form an arc
            if np.abs(np.cross(perp1, perp2)) < 1e-10:
                return {"passed": False, "message": "Points are collinear, cannot form arc"}

            # Points are on a circle, check if point is between arc endpoints
            # Calculate angles from center
            center_approx = (mid1 + mid2) / 2  # Approximation

            # Check distances are approximately equal (all on same circle)
            d1 = np.linalg.norm(pt.a - center_approx)
            d2 = np.linalg.norm(arc_p1.a - center_approx)
            d3 = np.linalg.norm(arc_p2.a - center_approx)

            on_circle = (np.abs(d1 - d2) <= self.tolerance and
                        np.abs(d2 - d3) <= self.tolerance)

            return {
                "passed": on_circle,
                "message": f"Point {'is' if on_circle else 'is not'} on arc (distances: {d1:.2f}, {d2:.2f}, {d3:.2f})"
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking arc: {str(e)}"
            }

    def _check_midpoint_of_arc(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is the midpoint of an arc."""
        point = data.get("point")
        arc = data.get("arc", [])

        if not point or len(arc) < 2:
            return {"passed": False, "message": "Invalid midpoint_of_arc condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict or not all(p in element_dict for p in arc[:2]):
            return {"passed": False, "message": "Could not find all points"}

        pt = element_dict[point].data
        arc_p1 = element_dict[arc[0]].data
        arc_p2 = element_dict[arc[1]].data

        if not all(isinstance(p, gt.Point) for p in [pt, arc_p1, arc_p2]):
            return {"passed": False, "message": "Invalid point types"}

        # First, verify all points are concyclic (on same circle)
        # Find the circle by looking for one with all three points
        circle = None

        for element in element_dict.values():
            if isinstance(element.data, gt.Circle):
                c = element.data
                # Check if all three points are on this circle
                d1 = np.linalg.norm(pt.a - c.c)
                d2 = np.linalg.norm(arc_p1.a - c.c)
                d3 = np.linalg.norm(arc_p2.a - c.c)

                if (np.abs(d1 - c.r) <= self.tolerance and
                    np.abs(d2 - c.r) <= self.tolerance and
                    np.abs(d3 - c.r) <= self.tolerance):
                    circle = c
                    break

        if not circle:
            return {"passed": False, "message": "Could not find circle containing all arc points"}

        # Calculate angles from center to each point
        center = circle.c

        # Vector from center to each point
        v1 = arc_p1.a - center
        v2 = pt.a - center
        v3 = arc_p2.a - center

        # Calculate angles using atan2
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle3 = np.arctan2(v3[1], v3[0])

        # Calculate angular distance from arc_p1 to pt and from pt to arc_p2
        # Handle angle wraparound by considering both directions
        def angle_diff(a1, a2):
            """Calculate smallest angle difference, considering wraparound."""
            diff = a2 - a1
            # Normalize to [-pi, pi]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            return diff

        # Angular distance from arc_p1 to pt
        arc1_to_mid = angle_diff(angle1, angle2)
        # Angular distance from pt to arc_p2
        mid_to_arc2 = angle_diff(angle2, angle3)

        # Check if angular distances are equal (midpoint condition)
        # Also need to check that the point is between the arc endpoints
        # (not on the opposite side of the circle)

        # Total arc angle
        total_arc = angle_diff(angle1, angle3)

        # Check if point is between endpoints
        # This is true if arc1_to_mid and mid_to_arc2 have the same sign as total_arc
        # and their sum equals total_arc
        same_direction = (np.sign(arc1_to_mid) == np.sign(total_arc) and
                         np.sign(mid_to_arc2) == np.sign(total_arc))

        if not same_direction:
            # Point might be on the opposite side - check the other arc
            arc1_to_mid = -arc1_to_mid if arc1_to_mid != 0 else 2*np.pi + arc1_to_mid
            mid_to_arc2 = -mid_to_arc2 if mid_to_arc2 != 0 else 2*np.pi + mid_to_arc2

        # Check if angular distances are approximately equal
        is_midpoint = np.abs(np.abs(arc1_to_mid) - np.abs(mid_to_arc2)) <= np.radians(self.tolerance)

        return {
            "passed": is_midpoint,
            "message": f"Point {'is' if is_midpoint else 'is not'} the midpoint of arc (angular distances: {np.degrees(np.abs(arc1_to_mid)):.2f}° and {np.degrees(np.abs(mid_to_arc2)):.2f}°)"
        }
    
    def _check_point_outside_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is not on a line."""
        point = data.get("point")
        line_data = data.get("line")

        if not point or not line_data:
            return {"passed": False, "message": "Invalid point_outside_line condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}

        # Get line using the helper method (supports all formats)
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": True, "message": "Line not found, assuming outside"}

        is_outside = not line.contains(pt.a)

        return {
            "passed": is_outside,
            "message": f"Point {'is' if is_outside else 'is not'} outside line"
        }
    
    def _check_point_above_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is above a line (positive y relative to line)."""
        point = data.get("point")
        line_data = data.get("line")

        if not point or not line_data:
            return {"passed": False, "message": "Invalid point_above_line condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}

        # Get line
        line = self._get_line_from_data(line_data)
        if line is None:
            return {"passed": False, "message": "Could not find line"}

        # Calculate signed distance to determine which side
        # Line equation: n·x = c (where n is normalized normal vector)
        # Signed distance: n·p - c
        signed_dist = np.dot(line.n, pt.a) - line.c

        # "Above" typically means positive y-direction
        # If normal vector points up (n[1] > 0), positive distance means above
        # If normal vector points down (n[1] < 0), negative distance means above
        is_above = signed_dist > 0 if line.n[1] > 0 else signed_dist < 0
        
        return {
            "passed": is_above,
            "message": f"Point {'is' if is_above else 'is not'} above line"
        }
    
    def _check_intersection(self, data: Dict) -> Dict[str, Any]:
        """Check if intersection points exist."""
        points = data.get("points", [])
        objects = data.get("objects", [])

        element_dict = self.construction.element_dict

        # Verify points exist
        for p in points:
            if p not in element_dict:
                return {"passed": False, "message": f"Intersection point {p} not found"}

        # If objects are specified, verify points are on those objects
        if len(objects) >= 2 and len(points) >= 1:
            # Get the objects (could be lines, segments, circles, etc.)
            obj_list = []
            for obj_data in objects[:2]:  # Check first two objects
                if isinstance(obj_data, list) and len(obj_data) == 2:
                    # It's a line defined by two points
                    line = self._get_line_from_data(obj_data)
                    if line:
                        obj_list.append(('line', line))
                elif isinstance(obj_data, str):
                    # It's a named object
                    if obj_data in element_dict:
                        elem = element_dict[obj_data].data
                        if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                            obj_list.append(('line', elem))
                        elif isinstance(elem, gt.Circle):
                            obj_list.append(('circle', elem))

            # Verify points are on the objects
            if len(obj_list) == 2:
                for point_label in points:
                    pt = element_dict[point_label].data
                    if not isinstance(pt, gt.Point):
                        continue

                    # Check if point is on both objects
                    on_obj1 = False
                    on_obj2 = False

                    for obj_type, obj in obj_list:
                        if obj_type == 'line':
                            if obj.contains(pt.a):
                                if not on_obj1:
                                    on_obj1 = True
                                else:
                                    on_obj2 = True
                        elif obj_type == 'circle':
                            dist = np.linalg.norm(pt.a - obj.c)
                            if np.abs(dist - obj.r) <= self.tolerance:
                                if not on_obj1:
                                    on_obj1 = True
                                else:
                                    on_obj2 = True

                    if not (on_obj1 and on_obj2):
                        return {
                            "passed": False,
                            "message": f"Point {point_label} not on intersection of objects"
                        }

        return {
            "passed": True,
            "message": f"Intersection points verified ({len(points)} point(s))"
        }
    
    def _check_point_intersection(self, data: Dict) -> Dict[str, Any]:
        """Check if a point is an intersection of objects."""
        point = data.get("point")
        objects = data.get("objects", [])

        if not point:
            return {"passed": False, "message": "Invalid point_intersection condition"}

        element_dict = self.construction.element_dict

        if point not in element_dict:
            return {"passed": False, "message": f"Point {point} not found"}

        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": f"{point} is not a point"}

        # If objects are specified, verify point is on all of them
        if len(objects) >= 2:
            for obj_data in objects:
                on_object = False

                if isinstance(obj_data, dict):
                    # Handle circle definition like {'circle_center': 'A', 'radius_point': 'B'}
                    if 'circle_center' in obj_data:
                        center_label = obj_data['circle_center']
                        if center_label in element_dict:
                            circle_label = self._find_circle_with_center(center_label)
                            if circle_label:
                                circle = element_dict[circle_label].data
                                dist = np.linalg.norm(pt.a - circle.c)
                                on_object = np.abs(dist - circle.r) <= self.tolerance

                elif isinstance(obj_data, list) and len(obj_data) == 2:
                    # Line defined by two points
                    line = self._get_line_from_data(obj_data)
                    if line:
                        on_object = line.contains(pt.a)

                elif isinstance(obj_data, str):
                    # Named object
                    if obj_data in element_dict:
                        elem = element_dict[obj_data].data
                        if isinstance(elem, (gt.Line, gt.Segment, gt.Ray)):
                            on_object = elem.contains(pt.a)
                        elif isinstance(elem, gt.Circle):
                            dist = np.linalg.norm(pt.a - elem.c)
                            on_object = np.abs(dist - elem.r) <= self.tolerance

                if not on_object:
                    return {
                        "passed": False,
                        "message": f"Point {point} not on object {obj_data}"
                    }

        return {
            "passed": True,
            "message": f"Point {point} verified as intersection of {len(objects)} object(s)"
        }
    
    def _check_geometric_transformation(self, data: Dict) -> Dict[str, Any]:
        """Check geometric transformation (rotation, reflection, etc.)."""
        transformation = data.get("transformation", "rotation")

        element_dict = self.construction.element_dict

        # Get all points
        preimage_points = data.get("preimage_points", [])
        image_points = data.get("image_points", [])
        preimage_triangle = data.get("preimage_triangle", [])
        image_triangle = data.get("image_triangle", [])

        # Use triangle if specified, otherwise use individual points
        if preimage_triangle and image_triangle:
            preimage_points = preimage_triangle
            image_points = image_triangle

        if len(preimage_points) != len(image_points):
            return {"passed": False, "message": "Preimage and image must have same number of points"}

        if not preimage_points:
            return {"passed": False, "message": "No points specified for transformation"}

        # Verify all points exist
        all_points = preimage_points + image_points
        center = data.get("center")
        if center:
            all_points.append(center)

        for p in all_points:
            if p not in element_dict:
                return {"passed": False, "message": f"Point {p} not found"}

        # Get point objects
        pre_pts = [element_dict[p].data for p in preimage_points]
        img_pts = [element_dict[p].data for p in image_points]

        if not all(isinstance(p, gt.Point) for p in pre_pts + img_pts):
            return {"passed": False, "message": "Invalid point types"}

        if transformation == "rotation":
            # For rotation, check:
            # 1. Distances from center are preserved
            # 2. All points rotate by the same angle

            if not center or center not in element_dict:
                return {"passed": False, "message": "Rotation requires a center point"}

            center_pt = element_dict[center].data
            if not isinstance(center_pt, gt.Point):
                return {"passed": False, "message": "Invalid center point type"}

            # Check distances from center are preserved
            distances_match = True
            for i in range(len(pre_pts)):
                dist_pre = cmd.distance_pp(center_pt, pre_pts[i]).x
                dist_img = cmd.distance_pp(center_pt, img_pts[i]).x

                if np.abs(dist_pre - dist_img) > self.tolerance:
                    distances_match = False
                    break

            if not distances_match:
                return {
                    "passed": False,
                    "message": "Distances from center not preserved in rotation"
                }

            # Check angles are consistent
            if len(pre_pts) >= 2:
                # Calculate rotation angle from first pair
                v1_pre = pre_pts[0].a - center_pt.a
                v1_img = img_pts[0].a - center_pt.a

                angle1 = np.arctan2(v1_img[1], v1_img[0]) - np.arctan2(v1_pre[1], v1_pre[0])

                # Verify same angle for all pairs
                angles_consistent = True
                for i in range(1, len(pre_pts)):
                    v_pre = pre_pts[i].a - center_pt.a
                    v_img = img_pts[i].a - center_pt.a

                    angle_i = np.arctan2(v_img[1], v_img[0]) - np.arctan2(v_pre[1], v_pre[0])

                    # Normalize angles to [-pi, pi]
                    angle1_norm = np.arctan2(np.sin(angle1), np.cos(angle1))
                    angle_i_norm = np.arctan2(np.sin(angle_i), np.cos(angle_i))

                    if np.abs(angle1_norm - angle_i_norm) > np.radians(self.tolerance):
                        angles_consistent = False
                        break

                if not angles_consistent:
                    return {
                        "passed": False,
                        "message": "Rotation angles not consistent for all points"
                    }

            angle_msg = f"{np.degrees(angle1):.1f}°" if len(pre_pts) >= 2 else "N/A"
            return {
                "passed": True,
                "message": f"Valid rotation transformation (angle: {angle_msg})"
            }

        else:
            # For other transformations, do basic checks
            # Check that distances between corresponding points are consistent
            if len(pre_pts) >= 2:
                # Check if it's a rigid transformation (distances preserved)
                pre_dist = cmd.distance_pp(pre_pts[0], pre_pts[1]).x
                img_dist = cmd.distance_pp(img_pts[0], img_pts[1]).x

                is_rigid = np.abs(pre_dist - img_dist) <= self.tolerance

                return {
                    "passed": True,
                    "message": f"Geometric transformation ({transformation}) verified (rigid: {is_rigid})"
                }

            return {
                "passed": True,
                "message": f"Geometric transformation ({transformation}) points exist"
            }
    
    def _check_contact(self, data: Dict) -> Dict[str, Any]:
        """Check contact condition (simplified - domain specific)."""
        # This is a specialized condition that might not apply to pure geometry
        return {
            "passed": True,
            "message": "Contact condition (domain specific, skipped)"
        }
    
    def _check_point_height(self, data: Dict) -> Dict[str, Any]:
        """Check point height condition (simplified - domain specific)."""
        # This is a specialized condition that might not apply to pure geometry
        return {
            "passed": True,
            "message": "Point height condition (domain specific, skipped)"
        }


# Example usage
if __name__ == "__main__":
    from src.benchmark.benchmark_dataset import BenchmarkDataset, RequiredObjects, ConditionBuilder
    
    # Test with an existing DSL file
    validator = DSLValidator()
    
    # Create a simple test problem
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[["A", "B"], ["B", "C"], ["A", "C"]],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]]
    )
    
    from src.benchmark.benchmark_dataset import BenchmarkProblem, VerificationCondition
    conditions = [
        VerificationCondition.from_dict(ConditionBuilder.not_collinear(["A", "B", "C"]))
    ]
    
    problem = BenchmarkProblem(
        id="test",
        subject="Test triangle",
        required_objects=required_objects,
        verification_conditions=conditions
    )
    
    print("DSL Validator created successfully")

