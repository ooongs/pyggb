#!/usr/bin/env python3
"""
DSL Validator for Geometry Benchmark
Validates DSL files against required objects and verification conditions.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from random_constr import Construction
from benchmark_dataset import BenchmarkProblem, VerificationCondition
import geo_types as gt
import commands as cmd


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
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "object_score": self.object_score,
            "condition_score": self.condition_score,
            "total_score": self.total_score,
            "missing_objects": self.missing_objects,
            "failed_conditions": self.failed_conditions,
            "error_message": self.error_message,
            "details": self.details
        }


class DSLValidator:
    """Validate DSL constructions against benchmark requirements."""
    
    def __init__(self, tolerance: float = 1e-2):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for geometric comparisons
        """
        self.tolerance = tolerance
        self.construction = Construction()
    
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
            
            return ValidationResult(
                success=success,
                object_score=object_score,
                condition_score=condition_score,
                total_score=total_score,
                missing_objects=missing_objects,
                failed_conditions=failed_conditions,
                details={
                    "object_details": object_result.get("details", {}),
                    "condition_details": condition_result.get("details", [])
                }
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
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )
    
    def _check_required_objects(self, required_objects) -> Dict[str, Any]:
        """
        Check if all required objects exist in the construction.
        Uses hybrid validation: explicit for polygons/circles, implicit for segments/lines.
        """
        element_dict = self.construction.element_dict
        
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
                found["polygons"].append(poly_points)
            else:
                # Relaxed: check if all points exist (polygon structure can be inferred)
                if all(p in element_dict for p in poly_points):
                    all_points = all(isinstance(element_dict[p].data, gt.Point) for p in poly_points)
                    if all_points:
                        # Check if points are not collinear (valid polygon)
                        if len(poly_points) == 3:
                            # Triangle: check non-collinearity
                            p1, p2, p3 = [element_dict[p].data for p in poly_points]
                            collinear = cmd.are_collinear_ppp(p1, p2, p3)
                            if not collinear.b:
                                found["polygons"].append(poly_points)
                            else:
                                missing["polygons"].append(poly_points)
                        else:
                            # Other polygons: accept if points exist
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
        """Find a line through two points in the construction."""
        element_dict = self.construction.element_dict
        
        if p1 not in element_dict or p2 not in element_dict:
            return None
        
        pt1 = element_dict[p1].data
        pt2 = element_dict[p2].data
        
        if not isinstance(pt1, gt.Point) or not isinstance(pt2, gt.Point):
            return None
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Line):
                line = element.data
                # Check if both points lie on the line
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
    
    def _points_match_polygon(self, points1: List[np.ndarray], 
                             points2: np.ndarray) -> bool:
        """Check if two sets of points match (considering rotation)."""
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
        
        # Try reverse order
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
            else:
                return {
                    "passed": False,
                    "message": f"Unknown condition type: {condition_type}"
                }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking condition: {str(e)}"
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
    
    def _check_parallel(self, data: Dict) -> Dict[str, Any]:
        """Check if two lines are parallel."""
        objects = data.get("objects", [])
        if len(objects) != 2:
            return {"passed": False, "message": "Parallel condition requires 2 lines"}
        
        line1 = self._get_line_from_points(objects[0][0], objects[0][1])
        line2 = self._get_line_from_points(objects[1][0], objects[1][1])
        
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
        
        line1 = self._get_line_from_points(objects[0][0], objects[0][1])
        line2 = self._get_line_from_points(objects[1][0], objects[1][1])
        
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
        
        # Check if angle matches expected value
        passed = np.abs(angle_degrees - expected_value) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Angle is {angle_degrees:.2f}°, expected {expected_value}° (tolerance {tolerance}°)"
        }
    
    def _check_angle_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if two angles are equal."""
        points_list = data.get("points", [])
        tolerance = data.get("tolerance", 1.0)
        
        if len(points_list) != 2:
            return {"passed": False, "message": "Angle equality requires 2 angles"}
        
        element_dict = self.construction.element_dict
        
        # Get first angle
        points1 = points_list[0]
        if len(points1) != 3 or not all(p in element_dict for p in points1):
            return {"passed": False, "message": "Could not find first angle points"}
        
        p1a, p1b, p1c = [element_dict[p].data for p in points1]
        if not all(isinstance(p, gt.Point) for p in [p1a, p1b, p1c]):
            return {"passed": False, "message": "Invalid point types in first angle"}
        
        angle1 = cmd.angle_ppp(p1a, p1b, p1c)
        
        # Get second angle
        points2 = points_list[1]
        if len(points2) != 3 or not all(p in element_dict for p in points2):
            return {"passed": False, "message": "Could not find second angle points"}
        
        p2a, p2b, p2c = [element_dict[p].data for p in points2]
        if not all(isinstance(p, gt.Point) for p in [p2a, p2b, p2c]):
            return {"passed": False, "message": "Invalid point types in second angle"}
        
        angle2 = cmd.angle_ppp(p2a, p2b, p2c)
        
        # Check equality
        result = cmd.are_congruent_aa(angle1, angle2)
        
        angle1_deg = np.degrees(angle1.angle)
        angle2_deg = np.degrees(angle2.angle)
        
        return {
            "passed": result.b,
            "message": f"Angles are {angle1_deg:.2f}° and {angle2_deg:.2f}°"
        }
    
    def _check_segment_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if two segments are equal in length."""
        segments = data.get("segments", [])
        
        if len(segments) != 2:
            return {"passed": False, "message": "Segment equality requires 2 segments"}
        
        element_dict = self.construction.element_dict
        
        # Get segments
        seg1_points = segments[0]
        seg2_points = segments[1]
        
        if not all(p in element_dict for p in seg1_points + seg2_points):
            return {"passed": False, "message": "Could not find all points"}
        
        # Calculate distances
        p1a, p1b = [element_dict[p].data for p in seg1_points]
        p2a, p2b = [element_dict[p].data for p in seg2_points]
        
        if not all(isinstance(p, gt.Point) for p in [p1a, p1b, p2a, p2b]):
            return {"passed": False, "message": "Invalid point types"}
        
        dist1 = cmd.distance_pp(p1a, p1b)
        dist2 = cmd.distance_pp(p2a, p2b)
        
        result = cmd.are_equal_mm(dist1, dist2)
        
        return {
            "passed": result.b,
            "message": f"Segments have lengths {dist1.x:.2f} and {dist2.x:.2f}"
        }
    
    def _check_collinear(self, data: Dict) -> Dict[str, Any]:
        """Check if points are collinear."""
        points = data.get("points", [])
        
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
        
        return {
            "passed": result.b,
            "message": f"Points {'are' if result.b else 'are not'} collinear"
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
        for line_points in lines:
            line = self._get_line_from_points(line_points[0], line_points[1])
            if line is None:
                return {"passed": False, "message": "Could not find all lines"}
            line_objs.append(line)
        
        result = cmd.are_concurrent_lll(*line_objs)
        
        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} concurrent"
        }
    
    def _check_point_on_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a line."""
        point = data.get("point")
        line_points = data.get("line", [])
        
        if not point or len(line_points) != 2:
            return {"passed": False, "message": "Invalid point_on_line condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}
        
        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}
        
        line = self._get_line_from_points(line_points[0], line_points[1])
        if line is None:
            return {"passed": False, "message": "Could not find line"}
        
        result = cmd.contained_by_pl(pt, line)
        
        return {
            "passed": result.b,
            "message": f"Point {'is' if result.b else 'is not'} on line"
        }
    
    def _check_point_on_circle(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a circle."""
        point = data.get("point")
        circle_center = data.get("circle_center")
        
        if not point or not circle_center:
            return {"passed": False, "message": "Invalid point_on_circle condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict or circle_center not in element_dict:
            return {"passed": False, "message": "Could not find point or circle"}
        
        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}
        
        # Find circle with this center
        circle_label = self._find_circle_with_center(circle_center)
        if not circle_label:
            return {"passed": False, "message": "Could not find circle"}
        
        circle = element_dict[circle_label].data
        result = cmd.contained_by_pc(pt, circle)
        
        return {
            "passed": result.b,
            "message": f"Point {'is' if result.b else 'is not'} on circle"
        }
    
    def _check_angle_bisector(self, data: Dict) -> Dict[str, Any]:
        """Check if a line bisects an angle."""
        line_points = data.get("line", [])
        angle_points = data.get("angle_points", [])
        
        if len(line_points) != 2 or len(angle_points) != 3:
            return {"passed": False, "message": "Invalid angle_bisector condition"}
        
        element_dict = self.construction.element_dict
        
        # Get angle vertex (middle point)
        vertex = angle_points[1]
        if vertex not in element_dict:
            return {"passed": False, "message": "Could not find angle vertex"}
        
        # Check if bisector line passes through vertex
        if vertex not in line_points:
            # Check if vertex lies on the line
            line = self._get_line_from_points(line_points[0], line_points[1])
            vertex_pt = element_dict[vertex].data
            if not isinstance(vertex_pt, gt.Point) or not line.contains(vertex_pt.a):
                return {"passed": False, "message": "Bisector doesn't pass through angle vertex"}
        
        # Get all points
        if not all(p in element_dict for p in angle_points):
            return {"passed": False, "message": "Could not find angle points"}
        
        p1, vertex_pt, p3 = [element_dict[p].data for p in angle_points]
        
        if not all(isinstance(p, gt.Point) for p in [p1, vertex_pt, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Get a point on the bisector (not the vertex)
        bisector_pt = None
        for lp in line_points:
            if lp != vertex and lp in element_dict:
                bisector_pt = element_dict[lp].data
                break
        
        if bisector_pt is None or not isinstance(bisector_pt, gt.Point):
            return {"passed": False, "message": "Could not find bisector point"}
        
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
        segment = data.get("segment", [])
        
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
        
        # Check if distance matches expected value
        passed = np.abs(actual_value - expected_value) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Distance is {actual_value:.2f}, expected {expected_value:.2f} (tolerance {tolerance:.2f})"
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


# Example usage
if __name__ == "__main__":
    from benchmark_dataset import BenchmarkDataset, RequiredObjects, ConditionBuilder
    
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
    
    from benchmark_dataset import BenchmarkProblem, VerificationCondition
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

