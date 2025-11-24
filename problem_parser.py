#!/usr/bin/env python3
"""
Problem Parser for Geometry Benchmark
Extracts geometric requirements from Chinese problem text using LLM.
"""

import json
import re
import os
from typing import Dict, List, Any, Optional


class ProblemParser:
    """Parse Chinese geometry problems to extract required objects and conditions."""
    
    def __init__(self, llm_api_function=None):
        """
        Initialize parser with optional LLM API function.
        
        Args:
            llm_api_function: Function that takes a prompt and returns LLM response
        """
        self.llm_api = llm_api_function
    
    def parse_problem(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """
        Parse a geometry problem and extract requirements.
        
        Args:
            problem_text: Chinese text describing the geometry problem
            problem_id: Optional problem identifier
            
        Returns:
            Dictionary with required_objects and verification_conditions
        """
        if self.llm_api:
            return self._parse_with_llm(problem_text, problem_id)
        else:
            # Fallback to rule-based parsing
            return self._parse_rule_based(problem_text, problem_id)
    
    def _parse_with_llm(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """Use LLM API to parse problem text."""
        prompt = self._create_parsing_prompt(problem_text)
        
        try:
            response = self.llm_api(prompt)
            
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            parsed_data = json.loads(response)
            
            result = {
                "id": problem_id or "unknown",
                "subject": problem_text,
                "required_objects": parsed_data.get("required_objects", {}),
                "verification_conditions": parsed_data.get("verification_conditions", [])
            }
            
            # Validate and auto-correct the parsed data
            result = self._validate_and_correct_parsed_data(result)
            
            return result
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to rule-based parsing")
            return self._parse_rule_based(problem_text, problem_id)
    
    def _parse_rule_based(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """
        Rule-based parsing as fallback or for manual annotation.
        Extracts basic patterns from Chinese text.
        """
        # Extract point names (uppercase letters)
        points = self._extract_points(problem_text)
        
        # Extract geometric relationships
        conditions = []
        
        # Check for parallel lines (平行 or ∥)
        if '平行' in problem_text or '∥' in problem_text:
            parallel_pairs = self._extract_parallel_lines(problem_text, points)
            for pair in parallel_pairs:
                conditions.append({
                    "type": "parallel",
                    "objects": pair
                })
        
        # Check for perpendicular (垂直 or ⊥)
        if '垂直' in problem_text or '⊥' in problem_text:
            perp_pairs = self._extract_perpendicular_lines(problem_text, points)
            for pair in perp_pairs:
                conditions.append({
                    "type": "perpendicular",
                    "objects": pair
                })
        
        # Check for angle conditions (角 and degree symbol °)
        angle_conditions = self._extract_angle_conditions(problem_text, points)
        conditions.extend(angle_conditions)
        
        # Check for bisector (平分)
        if '平分' in problem_text:
            bisector_conditions = self._extract_bisector_conditions(problem_text, points)
            conditions.extend(bisector_conditions)
        
        # Check for point on segment/line (在...上)
        if '在' in problem_text and '上' in problem_text:
            point_on_conditions = self._extract_point_on_conditions(problem_text, points)
            conditions.extend(point_on_conditions)
        
        # Check for midpoint (中点)
        if '中点' in problem_text:
            midpoint_conditions = self._extract_midpoint_conditions(problem_text, points)
            conditions.extend(midpoint_conditions)
        
        # Infer required objects from points
        required_objects = self._infer_objects_from_points(points, problem_text)
        
        return {
            "id": problem_id or "unknown",
            "subject": problem_text,
            "required_objects": required_objects,
            "verification_conditions": conditions
        }
    
    def _extract_points(self, text: str) -> List[str]:
        """Extract point names (typically uppercase letters in geometry)."""
        # Find all uppercase letters that appear to be point names
        points = re.findall(r'[A-Z]', text)
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        return unique_points
    
    def _extract_parallel_lines(self, text: str, points: List[str]) -> List[List[List[str]]]:
        """Extract parallel line pairs from text."""
        parallel_pairs = []
        
        # Pattern: AB∥CD or AB平行CD
        pattern = r'([A-Z]{2})[∥平行]+([A-Z]{2})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            line1 = list(match[0])  # Convert "AB" to ["A", "B"]
            line2 = list(match[1])  # Convert "CD" to ["C", "D"]
            parallel_pairs.append([line1, line2])
        
        return parallel_pairs
    
    def _extract_perpendicular_lines(self, text: str, points: List[str]) -> List[List[List[str]]]:
        """Extract perpendicular line pairs from text."""
        perp_pairs = []
        
        # Pattern: AB⊥CD or AB垂直CD
        pattern = r'([A-Z]{2})[⊥垂直]+([A-Z]{2})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            line1 = list(match[0])
            line2 = list(match[1])
            perp_pairs.append([line1, line2])
        
        return perp_pairs
    
    def _extract_angle_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract angle equality or value conditions."""
        conditions = []
        
        # Pattern: ∠ABC=50° or ∠ABC=∠DEF
        # First, find angle with specific value
        angle_value_pattern = r'∠([A-Z]{3})[=等于]*(\d+)°'
        matches = re.findall(angle_value_pattern, text)
        
        for match in matches:
            angle_points = list(match[0])  # "ABC" -> ["A", "B", "C"]
            value = float(match[1])
            # IMPORTANT: points must be nested array [["A", "B", "C"]] for angle_value
            conditions.append({
                "type": "angle_value",
                "points": [angle_points],  # Nested: [["A", "B", "C"]]
                "value": value
            })
        
        # Pattern: single point angle like ∠A=80° (in triangle context)
        # This is ambiguous - try to infer the full 3-point angle from context
        single_angle_pattern = r'∠([A-Z])[=等于]*(\d+)°'
        matches = re.findall(single_angle_pattern, text)
        
        for match in matches:
            vertex = match[0]
            value = float(match[1])
            
            # Try to find triangle containing this vertex
            triangle_pattern = r'三角形([A-Z]{3})'
            triangle_matches = re.findall(triangle_pattern, text)
            
            for tri in triangle_matches:
                tri_points = list(tri)
                if vertex in tri_points:
                    # Create 3-point angle with vertex in the middle
                    idx = tri_points.index(vertex)
                    # Get the other two points
                    other_points = [p for p in tri_points if p != vertex]
                    if len(other_points) == 2:
                        angle_points = [other_points[0], vertex, other_points[1]]
                        conditions.append({
                            "type": "angle_value",
                            "points": [angle_points],  # Nested: [[P1, vertex, P2]]
                            "value": value
                        })
                        break
        
        # Pattern: angle equality ∠ABC=∠DEF
        angle_eq_pattern = r'∠([A-Z]{3})[=等于]+∠([A-Z]{3})'
        matches = re.findall(angle_eq_pattern, text)
        
        for match in matches:
            angle1 = list(match[0])
            angle2 = list(match[1])
            conditions.append({
                "type": "angle_equality",
                "points": [angle1, angle2]
            })
        
        return conditions
    
    def _extract_bisector_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract angle bisector conditions."""
        conditions = []
        
        # Pattern: XY平分∠ABC (XY bisects angle ABC)
        pattern = r'([A-Z]{2})平分∠([A-Z]{3})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            bisector_line = list(match[0])
            angle_points = list(match[1])
            conditions.append({
                "type": "angle_bisector",
                "line": bisector_line,
                "angle_points": angle_points
            })
        
        return conditions
    
    def _extract_point_on_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract point-on-segment/line conditions."""
        conditions = []
        
        # Pattern: D在AB上 or D、E分别在AB、AC上
        # Match: X在YZ上
        pattern1 = r'([A-Z])在([A-Z]{2})上'
        matches = re.findall(pattern1, text)
        
        for match in matches:
            point = match[0]
            segment = list(match[1])
            conditions.append({
                "type": "point_on_segment",
                "point": point,
                "segment": segment
            })
        
        # Pattern: D、E分别在AB、AC上
        pattern2 = r'([A-Z])、([A-Z])分别在([A-Z]{2})、([A-Z]{2})上'
        matches = re.findall(pattern2, text)
        
        for match in matches:
            point1 = match[0]
            point2 = match[1]
            segment1 = list(match[2])
            segment2 = list(match[3])
            
            conditions.append({
                "type": "point_on_segment",
                "point": point1,
                "segment": segment1
            })
            conditions.append({
                "type": "point_on_segment",
                "point": point2,
                "segment": segment2
            })
        
        return conditions
    
    def _extract_midpoint_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract midpoint conditions."""
        conditions = []
        
        # Pattern: M是AB的中点 or M为AB中点
        pattern = r'([A-Z])[是为]([A-Z]{2})[的]?中点'
        matches = re.findall(pattern, text)
        
        for match in matches:
            point = match[0]
            segment = list(match[1])
            conditions.append({
                "type": "midpoint_of",
                "point": point,
                "segment": segment
            })
        
        return conditions
    
    def _validate_and_correct_parsed_data(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and auto-correct common mistakes in parsed data.
        
        Args:
            parsed_data: Parsed problem data
            
        Returns:
            Corrected parsed data
        """
        conditions = parsed_data.get("verification_conditions", [])
        corrected_conditions = []
        
        for condition in conditions:
            cond_type = condition.get("type", "")
            corrected = dict(condition)  # Copy
            
            # Fix angle_value: ensure points are nested
            if cond_type == "angle_value":
                points = condition.get("points", [])
                if points and not isinstance(points[0], list):
                    # Points are flat: ["A", "B", "C"] → [["A", "B", "C"]]
                    corrected["points"] = [points]
                    print(f"Auto-corrected angle_value: nested points list")
            
            # Fix angle_equality: ensure points are nested
            elif cond_type == "angle_equality":
                points = condition.get("points", [])
                if points and len(points) == 6:
                    # Flat list of 6 points → [[p1, p2, p3], [p4, p5, p6]]
                    corrected["points"] = [points[:3], points[3:]]
                    print(f"Auto-corrected angle_equality: split into two angles")
            
            # Validate parallel/perpendicular use "objects" field
            elif cond_type in ["parallel", "perpendicular"]:
                if "lines" in condition and "objects" not in condition:
                    corrected["objects"] = condition["lines"]
                    del corrected["lines"]
                    print(f"Auto-corrected {cond_type}: renamed 'lines' to 'objects'")
            
            # Validate all referenced points exist
            points_in_condition = self._extract_points_from_condition(corrected)
            all_points = parsed_data.get("required_objects", {}).get("points", [])
            
            # Add missing points to required_objects
            for point in points_in_condition:
                if point not in all_points:
                    all_points.append(point)
                    print(f"Auto-added missing point: {point}")
            
            corrected_conditions.append(corrected)
        
        parsed_data["verification_conditions"] = corrected_conditions
        return parsed_data
    
    def _extract_points_from_condition(self, condition: Dict) -> List[str]:
        """Extract all point labels from a condition."""
        points = []
        
        def extract_from_value(val):
            if isinstance(val, str) and len(val) == 1 and val.isupper():
                points.append(val)
            elif isinstance(val, list):
                for item in val:
                    extract_from_value(item)
            elif isinstance(val, dict):
                for v in val.values():
                    extract_from_value(v)
        
        for key, value in condition.items():
            if key != "type":
                extract_from_value(value)
        
        return points
    
    def _infer_objects_from_points(self, points: List[str], text: str) -> Dict[str, List]:
        """Infer required geometric objects from points and text."""
        required_objects = {
            "points": points,
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        # Find segments mentioned in text (e.g., AB, BC, CD)
        segment_pattern = r'([A-Z]{2})'
        potential_segments = re.findall(segment_pattern, text)
        
        # Filter to valid segments (both points exist)
        for seg in potential_segments:
            if len(seg) == 2 and seg[0] in points and seg[1] in points:
                seg_list = [seg[0], seg[1]]
                if seg_list not in required_objects["segments"]:
                    required_objects["segments"].append(seg_list)
        
        # Check for triangles (三角形)
        if '三角形' in text:
            triangle_pattern = r'三角形([A-Z]{3})'
            matches = re.findall(triangle_pattern, text)
            for match in matches:
                if len(match) == 3:
                    triangle = list(match)
                    if triangle not in required_objects["polygons"]:
                        required_objects["polygons"].append(triangle)
        
        # Check for quadrilaterals (四边形)
        if '四边形' in text:
            quad_pattern = r'四边形([A-Z]{4})'
            matches = re.findall(quad_pattern, text)
            for match in matches:
                if len(match) == 4:
                    quad = list(match)
                    if quad not in required_objects["polygons"]:
                        required_objects["polygons"].append(quad)
        
        # Check for circles (圆 or ⊙)
        if '圆' in text or '⊙' in text:
            circle_pattern = r'[圆⊙]([A-Z])'
            matches = re.findall(circle_pattern, text)
            for match in matches:
                if match not in required_objects["circles"]:
                    required_objects["circles"].append({"center": match})
        
        return required_objects
    
    def _create_parsing_prompt(self, problem_text: str) -> str:
        """Create a prompt for LLM to parse geometry problem."""
        prompt = f"""Parse the following Chinese geometry problem and extract:
1. Required geometric objects (points, segments, lines, circles, polygons)
2. Geometric conditions that must be verified (parallel, perpendicular, angles, etc.)

Problem: {problem_text}

Return a JSON object with this structure:
{{
  "required_objects": {{
    "points": ["A", "B", "C", ...],
    "segments": [["A", "B"], ["B", "C"], ...],
    "lines": [["A", "B"], ["C", "D"], ...],
    "circles": [{{"center": "O", "radius_point": "A"}}, ...],
    "polygons": [["A", "B", "C"], ...]
  }},
  "verification_conditions": [
    {{"type": "parallel", "objects": [["A", "B"], ["C", "D"]]}},
    {{"type": "perpendicular", "objects": [["A", "B"], ["C", "D"]]}},
    {{"type": "angle_value", "points": [["A", "B", "C"]], "value": 50}},
    {{"type": "angle_equality", "points": [["A", "B", "C"], ["D", "E", "F"]]}},
    {{"type": "segment_equality", "segments": [["A", "B"], ["C", "D"]]}},
    {{"type": "collinear", "points": ["A", "B", "C"]}},
    {{"type": "not_collinear", "points": ["A", "B", "C"]}},
    {{"type": "point_on_line", "point": "D", "line": ["A", "B"]}},
    {{"type": "point_on_segment", "point": "D", "segment": ["A", "B"]}},
    {{"type": "point_on_circle", "point": "A", "circle_center": "O"}},
    {{"type": "angle_bisector", "line": ["E", "G"], "angle_points": ["B", "E", "F"]}},
    {{"type": "midpoint_of", "point": "M", "segment": ["A", "B"]}},
    {{"type": "distance_equals", "segment": ["A", "B"], "value": 10.0}},
    {{"type": "triangle_valid", "points": ["A", "B", "C"]}},
    {{"type": "concyclic", "points": ["A", "B", "C", "D"]}},
    {{"type": "concurrent", "lines": [["A", "B"], ["C", "D"], ["E", "F"]]}}
  ]
}}

CRITICAL RULES FOR CONDITION TYPES:

1. **angle_value**: ALWAYS use 3 points in nested format [["P1", "P2", "P3"]] where P2 is the vertex.
   - ✓ CORRECT: {{"type": "angle_value", "points": [["B", "A", "C"]], "value": 80}}
   - ✗ WRONG: {{"type": "angle_value", "points": ["B", "A", "C"], "value": 80}}
   - If text says "∠A=80°" in triangle ABC, use [["B", "A", "C"]] or [["C", "A", "B"]] with A in the middle.

2. **parallel**: Use "objects" field with two 2-point lines.
   - ✓ CORRECT: {{"type": "parallel", "objects": [["A", "B"], ["C", "D"]]}}
   - Chinese: "AB∥CD" or "AB平行CD"

3. **perpendicular**: Use "objects" field with two 2-point lines.
   - ✓ CORRECT: {{"type": "perpendicular", "objects": [["A", "B"], ["C", "D"]]}}
   - Chinese: "AB⊥CD" or "AB垂直CD"

4. **point_on_segment**: When text says "D在AB上" (D is on AB).
   - ✓ CORRECT: {{"type": "point_on_segment", "point": "D", "segment": ["A", "B"]}}
   - Use this instead of "point_on_line" when the point must be BETWEEN the endpoints.

5. **angle_bisector**: When a line bisects an angle.
   - ✓ CORRECT: {{"type": "angle_bisector", "line": ["E", "G"], "angle_points": ["B", "E", "F"]}}
   - Chinese: "EG平分∠BEF"

6. **midpoint_of**: When a point is the midpoint of a segment.
   - ✓ CORRECT: {{"type": "midpoint_of", "point": "M", "segment": ["A", "B"]}}
   - Chinese: "M是AB的中点" or "M为AB中点"

7. **triangle_valid**: For ensuring non-degenerate triangles (usually implicit).
   - ✓ CORRECT: {{"type": "triangle_valid", "points": ["A", "B", "C"]}}

STEP-BY-STEP PARSING GUIDE:

Step 1: Extract all point names (uppercase letters like A, B, C, D, E, etc.)

Step 2: Identify geometric shapes:
- Triangle: "三角形ABC" → polygon ["A", "B", "C"]
- Quadrilateral: "四边形ABCD" → polygon ["A", "B", "C", "D"]
- Circle: "圆O" or "⊙O" → circle with center "O"

Step 3: Extract segments and lines mentioned:
- Any two consecutive points in a shape form a segment
- Lines explicitly mentioned (e.g., "直线AB")

Step 4: Parse conditions:
- Parallel: "∥" or "平行"
- Perpendicular: "⊥" or "垂直"
- Angles: "∠ABC=50°" → angle_value with proper nesting
- Points on segments/lines: "D在AB上" or "D、E分别在AB、AC上"
- Bisectors: "平分"
- Equal segments: "AB=CD"

Step 5: Validate JSON structure:
- All angle_value conditions must have nested points: [["A", "B", "C"]]
- All point references must be in the points list
- All segment/line references must use valid point pairs

COMMON MISTAKES TO AVOID:
- ✗ {{"type": "angle_value", "points": ["A", "B", "C"]}} // Missing nested list
- ✗ {{"type": "parallel", "lines": [...]}} // Use "objects" not "lines"
- ✗ {{"type": "point_on_line", ...}} when point is on segment // Use "point_on_segment"
- ✗ Forgetting to include all points mentioned in the problem

Only return the JSON object, no other text or markdown formatting.
"""
        return prompt
    
    def parse_from_json(self, json_file: str) -> Dict[str, Any]:
        """
        Parse problem from JSON file (like GeoQA3 format).
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Parsed problem data
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problem_text = data.get('subject', '')
        problem_id = str(data.get('id', 'unknown'))
        
        return self.parse_problem(problem_text, problem_id)
    
    def save_to_benchmark_format(self, parsed_data: Dict[str, Any], output_file: str):
        """Save parsed data to benchmark format JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)


def create_openai_api_function(model: str = "gpt-4o-mini", api_key: str = None):
    """
    Create an OpenAI API function for parsing.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        
    Returns:
        Function that takes a prompt and returns LLM response
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    def api_function(prompt: str) -> str:
        """Call OpenAI API with the prompt."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a geometry problem parser. You extract geometric objects and conditions from Chinese geometry problems and return them as JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    return api_function


# Example usage
if __name__ == "__main__":
    # Example: Parse the problem from 1.json
    
    # Option 1: Rule-based parsing (default)
    parser = ProblemParser()
    
    # Option 2: With OpenAI LLM (uncomment to use)
    # api_key = os.getenv("OPENAI_API_KEY")
    # if api_key:
    #     llm_function = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
    #     parser = ProblemParser(llm_api_function=llm_function)
    #     print("Using OpenAI API for parsing")
    # else:
    #     print("OPENAI_API_KEY not found, using rule-based parsing")
    
    # Test with the example problem text
    problem_text = "如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()"
    
    result = parser.parse_problem(problem_text, problem_id="1")
    
    print("Parsed Problem:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

