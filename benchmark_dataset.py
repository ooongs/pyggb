#!/usr/bin/env python3
"""
Benchmark Dataset Management
Handles loading, saving, and managing geometry benchmark datasets.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RequiredObjects:
    """Structure for required geometric objects."""
    points: List[str]
    segments: List[List[str]]
    lines: List[List[str]]
    circles: List[Dict[str, Any]]
    polygons: List[List[str]]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RequiredObjects':
        return cls(
            points=data.get('points', []),
            segments=data.get('segments', []),
            lines=data.get('lines', []),
            circles=data.get('circles', []),
            polygons=data.get('polygons', [])
        )


@dataclass
class VerificationCondition:
    """Structure for a single verification condition."""
    type: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        result = {"type": self.type}
        result.update(self.data)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationCondition':
        condition_type = data.pop('type')
        return cls(type=condition_type, data=data)


@dataclass
class BenchmarkProblem:
    """Structure for a single benchmark problem."""
    id: str
    subject: str
    required_objects: RequiredObjects
    verification_conditions: List[VerificationCondition]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "required_objects": self.required_objects.to_dict(),
            "verification_conditions": [c.to_dict() for c in self.verification_conditions],
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkProblem':
        required_objects = RequiredObjects.from_dict(data.get('required_objects', {}))
        conditions = [
            VerificationCondition.from_dict(c) 
            for c in data.get('verification_conditions', [])
        ]
        return cls(
            id=data.get('id', 'unknown'),
            subject=data.get('subject', ''),
            required_objects=required_objects,
            verification_conditions=conditions,
            metadata=data.get('metadata')
        )


class BenchmarkDataset:
    """Manage a collection of benchmark problems."""
    
    # Supported condition types (must match dsl_validator.py)
    SUPPORTED_CONDITION_TYPES = {
        "parallel", "perpendicular", "angle_value", "angle_equality",
        "segment_equality", "collinear", "not_collinear", "concyclic",
        "concurrent", "point_on_line", "point_on_circle", "angle_bisector",
        "point_on_segment", "midpoint_of", "distance_equals",
        "triangle_valid", "point_between", "concentric_circles"
    }
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset directory or JSON file
        """
        self.problems: List[BenchmarkProblem] = []
        self.dataset_path = dataset_path
        self.metadata: Dict[str, Any] = {}  # Metadata from new dataset format
        
        if dataset_path and os.path.exists(dataset_path):
            self.load(dataset_path)
    
    def load(self, path: str):
        """
        Load benchmark dataset from file or directory.
        
        Args:
            path: Path to JSON file or directory containing JSON files
        """
        if os.path.isfile(path):
            # Load single file
            self._load_file(path)
        elif os.path.isdir(path):
            # Load all JSON files in directory
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    filepath = os.path.join(path, filename)
                    self._load_file(filepath)
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    def _load_file(self, filepath: str):
        """Load a single JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle multiple formats:
        # 1. New format: {"metadata": {...}, "problems": [...]}
        # 2. Old format (list): [problem1, problem2, ...]
        # 3. Single problem: {problem}
        
        if isinstance(data, dict):
            if "problems" in data:
                # New format with metadata and problems array
                problems_list = data.get("problems", [])
                self.metadata = data.get("metadata", {})
                for item in problems_list:
                    self.problems.append(BenchmarkProblem.from_dict(item))
            else:
                # Single problem dict
                self.problems.append(BenchmarkProblem.from_dict(data))
        elif isinstance(data, list):
            # Old format: list of problems
            for item in data:
                self.problems.append(BenchmarkProblem.from_dict(item))
    
    def save(self, output_path: str, single_file: bool = True):
        """
        Save benchmark dataset.
        
        Args:
            output_path: Output file or directory path
            single_file: If True, save all problems to single JSON file
        """
        if single_file:
            # Save all problems to single file
            data = [p.to_dict() for p in self.problems]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # Save each problem to separate file
            os.makedirs(output_path, exist_ok=True)
            for problem in self.problems:
                filename = f"{problem.id}.json"
                filepath = os.path.join(output_path, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(problem.to_dict(), f, ensure_ascii=False, indent=2)
    
    def add_problem(self, problem: BenchmarkProblem):
        """Add a problem to the dataset."""
        self.problems.append(problem)
    
    def get_problem(self, problem_id: str) -> Optional[BenchmarkProblem]:
        """Get a problem by ID."""
        for problem in self.problems:
            if problem.id == problem_id:
                return problem
        return None
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __iter__(self):
        return iter(self.problems)
    
    def __getitem__(self, idx: int) -> BenchmarkProblem:
        return self.problems[idx]
    
    def validate_compatibility(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate dataset compatibility with validator.
        
        Args:
            verbose: Print detailed report
            
        Returns:
            Dictionary with validation results
        """
        unsupported_conditions = {}
        total_conditions = 0
        problems_with_issues = []
        
        for problem in self.problems:
            problem_issues = []
            
            for condition in problem.verification_conditions:
                total_conditions += 1
                cond_type = condition.type
                
                if cond_type not in self.SUPPORTED_CONDITION_TYPES:
                    if cond_type not in unsupported_conditions:
                        unsupported_conditions[cond_type] = []
                    unsupported_conditions[cond_type].append(problem.id)
                    problem_issues.append(cond_type)
            
            if problem_issues:
                problems_with_issues.append({
                    "id": problem.id,
                    "unsupported_types": problem_issues
                })
        
        is_compatible = len(unsupported_conditions) == 0
        
        result = {
            "compatible": is_compatible,
            "total_problems": len(self.problems),
            "total_conditions": total_conditions,
            "problems_with_issues": len(problems_with_issues),
            "unsupported_condition_types": unsupported_conditions,
            "problem_details": problems_with_issues
        }
        
        if verbose:
            print("\n" + "="*70)
            print("DATASET VALIDATION REPORT")
            print("="*70)
            print(f"Total Problems: {result['total_problems']}")
            print(f"Total Conditions: {result['total_conditions']}")
            print(f"Compatible: {'✓ YES' if is_compatible else '✗ NO'}")
            
            if not is_compatible:
                print(f"\nProblems with Issues: {result['problems_with_issues']}")
                print("\nUnsupported Condition Types:")
                for cond_type, problem_ids in unsupported_conditions.items():
                    print(f"  - {cond_type}: {len(problem_ids)} occurrences")
                    print(f"    Problems: {', '.join(problem_ids[:5])}" + 
                          (" ..." if len(problem_ids) > 5 else ""))
                
                print("\nSuggested Fixes:")
                print("1. Add missing condition handlers to dsl_validator.py")
                print("2. Update SUPPORTED_CONDITION_TYPES in benchmark_dataset.py")
                print("3. Or re-parse dataset with updated problem_parser.py")
            else:
                print("\n✓ All condition types are supported!")
            
            print("="*70)
        
        return result


class ConditionBuilder:
    """Helper class to build verification conditions programmatically."""
    
    @staticmethod
    def parallel(line1: List[str], line2: List[str]) -> Dict:
        """Create a parallel lines condition."""
        return {
            "type": "parallel",
            "objects": [line1, line2]
        }
    
    @staticmethod
    def perpendicular(line1: List[str], line2: List[str]) -> Dict:
        """Create a perpendicular lines condition."""
        return {
            "type": "perpendicular",
            "objects": [line1, line2]
        }
    
    @staticmethod
    def angle_value(points: List[str], value: float, tolerance: float = 0.01) -> Dict:
        """
        Create an angle value condition.
        
        Args:
            points: Three points defining the angle [P1, P2, P3] where P2 is vertex
            value: Expected angle value in degrees
            tolerance: Tolerance for angle comparison (default 0.01 degrees)
        """
        return {
            "type": "angle_value",
            "points": [points],
            "value": value,
            "tolerance": tolerance
        }
    
    @staticmethod
    def angle_equality(angle1_points: List[str], angle2_points: List[str], 
                       tolerance: float = 0.01) -> Dict:
        """
        Create an angle equality condition.
        
        Args:
            angle1_points: Three points for first angle
            angle2_points: Three points for second angle
            tolerance: Tolerance for comparison
        """
        return {
            "type": "angle_equality",
            "points": [angle1_points, angle2_points],
            "tolerance": tolerance
        }
    
    @staticmethod
    def segment_equality(seg1: List[str], seg2: List[str], tolerance: float = 0.01) -> Dict:
        """Create a segment equality condition."""
        return {
            "type": "segment_equality",
            "segments": [seg1, seg2],
            "tolerance": tolerance
        }
    
    @staticmethod
    def collinear(points: List[str]) -> Dict:
        """Create a collinearity condition."""
        return {
            "type": "collinear",
            "points": points
        }
    
    @staticmethod
    def not_collinear(points: List[str]) -> Dict:
        """Create a non-collinearity condition (for valid triangles)."""
        return {
            "type": "not_collinear",
            "points": points
        }
    
    @staticmethod
    def concyclic(points: List[str]) -> Dict:
        """Create a concyclic condition (four points on same circle)."""
        return {
            "type": "concyclic",
            "points": points
        }
    
    @staticmethod
    def concurrent(lines: List[List[str]]) -> Dict:
        """Create a concurrent lines condition (three lines meet at a point)."""
        return {
            "type": "concurrent",
            "lines": lines
        }
    
    @staticmethod
    def point_on_line(point: str, line: List[str]) -> Dict:
        """Create a point-on-line condition."""
        return {
            "type": "point_on_line",
            "point": point,
            "line": line
        }
    
    @staticmethod
    def point_on_circle(point: str, circle_center: str) -> Dict:
        """Create a point-on-circle condition."""
        return {
            "type": "point_on_circle",
            "point": point,
            "circle_center": circle_center
        }
    
    @staticmethod
    def angle_bisector(line: List[str], angle_points: List[str]) -> Dict:
        """
        Create angle bisector condition.
        
        Args:
            line: Two points defining the bisector line
            angle_points: Three points defining the angle being bisected
        """
        return {
            "type": "angle_bisector",
            "line": line,
            "angle_points": angle_points
        }
    
    @staticmethod
    def point_on_segment(point: str, segment: List[str]) -> Dict:
        """
        Create point-on-segment condition.
        
        Args:
            point: Point label
            segment: Two points defining the segment
        """
        return {
            "type": "point_on_segment",
            "point": point,
            "segment": segment
        }
    
    @staticmethod
    def midpoint_of(point: str, segment: List[str]) -> Dict:
        """
        Create midpoint condition.
        
        Args:
            point: Point that should be the midpoint
            segment: Two points defining the segment
        """
        return {
            "type": "midpoint_of",
            "point": point,
            "segment": segment
        }
    
    @staticmethod
    def distance_equals(segment: List[str], value: float, tolerance: float = 0.01) -> Dict:
        """
        Create distance equals condition.
        
        Args:
            segment: Two points defining the segment
            value: Expected distance value
            tolerance: Tolerance for comparison (default 0.01)
        """
        return {
            "type": "distance_equals",
            "segment": segment,
            "value": value,
            "tolerance": tolerance
        }
    
    @staticmethod
    def triangle_valid(points: List[str]) -> Dict:
        """
        Create valid triangle condition (non-degenerate).
        
        Args:
            points: Three points that should form a valid triangle
        """
        return {
            "type": "triangle_valid",
            "points": points
        }
    
    @staticmethod
    def point_between(point: str, endpoints: List[str]) -> Dict:
        """
        Create point-between condition.
        
        Args:
            point: Point that should be between endpoints
            endpoints: Two points defining the line segment
        """
        return {
            "type": "point_between",
            "point": point,
            "endpoints": endpoints
        }
    
    @staticmethod
    def concentric_circles(centers: List[str]) -> Dict:
        """
        Create concentric circles condition.
        
        Args:
            centers: List of circle center labels (should be same point)
        """
        return {
            "type": "concentric_circles",
            "centers": centers
        }


# Example usage and test
if __name__ == "__main__":
    # Create a sample problem
    required_objects = RequiredObjects(
        points=["A", "B", "C", "D", "E", "F", "G"],
        segments=[["A", "B"], ["C", "D"], ["E", "F"]],
        lines=[["A", "B"], ["C", "D"]],
        circles=[],
        polygons=[]
    )
    
    conditions = [
        VerificationCondition.from_dict(ConditionBuilder.parallel(["A", "B"], ["C", "D"])),
        VerificationCondition.from_dict(ConditionBuilder.angle_value(["B", "E", "F"], 130.0)),
        VerificationCondition.from_dict(ConditionBuilder.angle_bisector(["E", "G"], ["B", "E", "F"])),
    ]
    
    problem = BenchmarkProblem(
        id="test_1",
        subject="如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()",
        required_objects=required_objects,
        verification_conditions=conditions,
        metadata={"source": "GeoQA3", "difficulty": "medium"}
    )
    
    # Create dataset and add problem
    dataset = BenchmarkDataset()
    dataset.add_problem(problem)
    
    # Save to file
    output_file = "test_benchmark.json"
    dataset.save(output_file)
    print(f"Saved test benchmark to {output_file}")
    
    # Load it back
    loaded_dataset = BenchmarkDataset(output_file)
    print(f"Loaded {len(loaded_dataset)} problems")
    
    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)

