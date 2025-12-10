#!/usr/bin/env python3
"""
Problem Parser for Geometry Benchmark
Extracts geometric requirements from Chinese problem text using LLM.

Features:
1. Two-stage LLM processing:
   - Stage 1: Validate problem suitability + clean text (remove non-construction content)
   - Stage 2: Extract objects/conditions + rate construction difficulty
2. Advanced filtering for ambiguous problems:
   - Undefined points (e.g., ∠E without E's position)
   - Incomplete angle definitions (e.g., ∠D instead of ∠BDC)
   - Numbered angles (∠1, ∠2)
3. Difficulty rating from CONSTRUCTION perspective (not problem-solving)
4. Batch processing with incremental saving

Usage:
    from problem_parser import ProblemParser, create_openai_api_function
    
    llm_func = create_openai_api_function(model="gpt-4o-mini")
    parser = ProblemParser(llm_api_function=llm_func)
    
    # Parse single problem (two API calls: validate + parse)
    result = parser.parse_problem(problem_text, problem_id="1")
"""

import json
import re
import os
from typing import Dict, List, Any, Optional, Union, Tuple


class ProblemParser:
    """Parse Chinese geometry problems to extract required objects and conditions."""
    
    # Problem classification categories
    PROBLEM_CATEGORIES = [
        "Basic Constructions",
        "Circle Properties & Constructions",
        "Geometric Transformations",
        "Triangle Properties & Constructions",
        "Applications of Geometric Theorems",
        "Polygon Properties & Constructions",
        "Measurement & Ratios",
        "Locus Constructions",
        "Angle Relationships",
        "Similarity & Congruence"
    ]
    
    def __init__(self, llm_api_function=None):
        """
        Initialize parser with optional LLM API function.
        
        Args:
            llm_api_function: Function that takes a prompt and returns LLM response
        """
        self.llm_api = llm_api_function
    
    def has_ambiguous_references(self, problem_text: str) -> bool:
        """
        Check if problem has obvious ambiguous references like ∠1, ∠2.
        This is a quick pre-check before LLM validation.
        
        Args:
            problem_text: Problem text to check
            
        Returns:
            True if problem has ambiguous references, False otherwise
        """
        # Pattern for numbered angles: ∠1, ∠2, etc.
        ambiguous_patterns = [
            r'∠\d+',  # ∠1, ∠2, etc.
            r'∠[①②③④⑤⑥⑦⑧⑨⑩]',  # Circled numbers
        ]
        
        for pattern in ambiguous_patterns:
            if re.search(pattern, problem_text):
                return True
        
        return False
    
    def _basic_clean_text(self, problem_text: str) -> str:
        """
        Basic text cleaning (without LLM) - removes obvious non-construction content.
        
        Args:
            problem_text: Original problem text
            
        Returns:
            Cleaned problem text
        """
        text = problem_text
        
        # Remove score patterns like (3分), （5分）
        text = re.sub(r'[（\(]\s*\d+\s*分\s*[）\)]', '', text)
        
        # Remove common diagram reference phrases
        phrases_to_remove = [
            r'如图所?示[,，、]?',
            r'如图[,，、]?',
            r'图中[,，、]?',
            r'观察图[,，、]?',
        ]
        
        for pattern in phrases_to_remove:
            text = re.sub(pattern, '', text)
        
        # Remove question parts with parentheses
        question_patterns = [
            r'[,，、。]?则[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?那么[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?求[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?计算[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?证明[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?判断[^,，。]*[是为]?\s*\([^\)]*\)',
            r'[,，、。]?下列[^,，。]*正确的[是为]?\s*\([^\)]*\)',
        ]
        
        for pattern in question_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove trailing question patterns
        text = re.sub(r'[,，、。]?则[^。]*$', '', text)
        text = re.sub(r'[,，、。]?那么[^。]*$', '', text)
        text = re.sub(r'[,，、。]?求[^。]*$', '', text)
        
        # Clean up punctuation and spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^[,，、。\s]+', '', text)
        text = re.sub(r'[,，、。\s]+$', '', text)
        
        return text.strip()
    
    def validate_and_clean_problem(self, problem_text: str, problem_id: str = None) -> Tuple[bool, str, str]:
        """
        Stage 1 API Call: Validate problem suitability and clean text.
        
        Checks for:
        - Undefined points (e.g., ∠E without E's position defined)
        - Incomplete angle definitions (e.g., ∠D instead of ∠BDC)
        - Points not on defined lines/segments
        - All geometric objects properly defined
        
        Args:
            problem_text: Original problem text
            problem_id: Problem identifier
            
        Returns:
            Tuple of (is_valid, cleaned_text, rejection_reason)
        """
        if not self.llm_api:
            # Without LLM, just do basic cleaning
            cleaned = self._basic_clean_text(problem_text)
            return (True, cleaned, "")
        
        prompt = self._create_validation_prompt(problem_text)
        
        try:
            response = self.llm_api(prompt)
            response = response.strip()
            
            # Extract JSON from response
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            
            is_valid = result.get("is_valid", False)
            cleaned_text = result.get("cleaned_text", "")
            rejection_reason = result.get("rejection_reason", "")
            
            if not is_valid:
                print(f"Problem {problem_id} rejected: {rejection_reason}")
            
            return (is_valid, cleaned_text, rejection_reason)
            
        except Exception as e:
            print(f"Validation failed for {problem_id}: {e}")
            # Fallback to basic cleaning
            cleaned = self._basic_clean_text(problem_text)
            return (True, cleaned, "")
    
    def _create_validation_prompt(self, problem_text: str) -> str:
        """Create prompt for Stage 1: Validation and cleaning."""
        prompt = f"""Analyze this Chinese geometry problem for GEOMETRIC CONSTRUCTION suitability.

Problem: {problem_text}

Your task:
1. Determine if this problem can be used for geometric figure construction
2. Remove non-construction content (questions, scores, "如图", etc.)
3. Keep ONLY the geometric setup conditions

**REJECTION CRITERIA** (return is_valid: false if ANY apply):

1. **Undefined Points**: A point is mentioned but its position is not defined
   - ❌ "∠E=40°" - E's position is unknown (which line is E on?)
   - ❌ "∠BDC=30°" - D's position is not defined (D is on which line?)
   - ✓ "D在AB上,∠BDC=30°" - D is defined as being on AB

2. **Ambiguous Angles**: Single-letter angle without context
   - ❌ "∠D=26°" - Could mean ∠ADB, ∠BDC, ∠ADC, etc.
   - ❌ "∠E=35°" - Which angle at E?
   - ✓ "∠ABC=50°" - Clear 3-point angle
   - ✓ "在△ABC中,∠A=60°" - ∠A is ∠BAC in context of triangle ABC

3. **Numbered Angles**: References like ∠1, ∠2, ∠① require diagram
   - ❌ "∠1=30°,∠2=45°"

4. **Incomplete Constraints**: Not enough information to construct
   - ❌ "AB∥CD,∠E=40°" - E is not positioned, angle not constructible

5. **Pure Calculation Problems**: No actual construction, just asking for values
   - ❌ Problems that only ask to calculate without defining constructible geometry

**CLEANING RULES** (remove these from text):

1. Score indicators: "(3分)", "（5分）"
2. Diagram references: "如图", "如图所示", "图中"
3. Questions: "则∠AOB的大小是()", "那么∠BOD为()", "求...的值"
4. Proof requests: "证明...", "判断..."
5. Answer choices: "A. 30° B. 45° C. 60°"
6. Any text after "则", "那么", "求", "证明"

**KEEP in cleaned_text**:
- Shape definitions: "在△ABC中", "四边形ABCD"
- Point positions: "D在AB上", "E是BC的中点"
- Measurements: "AB=5", "∠ABC=60°"
- Relationships: "AB∥CD", "AB⊥BC"

Return JSON:
{{
    "is_valid": true/false,
    "cleaned_text": "cleaned problem text with only construction conditions",
    "rejection_reason": "reason if invalid, empty string if valid"
}}

Only return the JSON, no other text."""
        return prompt
    
    def parse_problem(self, problem_text: str, problem_id: str = None, 
                     skip_ambiguous: bool = True) -> Optional[Dict[str, Any]]:
        """
        Parse a geometry problem using two-stage LLM processing.
        
        Stage 1: Validate problem suitability + clean text
        Stage 2: Extract objects/conditions + rate difficulty
        
        Args:
            problem_text: Chinese text describing the geometry problem
            problem_id: Optional problem identifier
            skip_ambiguous: If True, skip problems with obvious ambiguous refs (∠1, ∠2)
            
        Returns:
            Dictionary with parsed data, or None if problem is not suitable
        """
        original_text = problem_text
        
        # Quick pre-check for numbered angles (no API needed)
        if skip_ambiguous and self.has_ambiguous_references(problem_text):
            print(f"Skipping problem {problem_id}: contains numbered angles (∠1, ∠2)")
            return None
        
        # Stage 1: Validate and clean (first API call)
        is_valid, cleaned_text, rejection_reason = self.validate_and_clean_problem(
            problem_text, problem_id
        )
        
        if not is_valid:
            return None
        
        if not cleaned_text or len(cleaned_text.strip()) < 5:
            print(f"Skipping problem {problem_id}: no constructible content after cleaning")
            return None
        
        # Stage 2: Parse objects/conditions and rate difficulty (second API call)
        if self.llm_api:
            result = self._parse_and_rate_with_llm(cleaned_text, problem_id)
        else:
            result = self._parse_rule_based(cleaned_text, problem_id)
            result["difficulty"] = 3  # Default
            result["category"] = "Unknown"
        
        # Add text info
        result["original_text"] = original_text
        result["cleaned_text"] = cleaned_text
        
        return result
    
    def _parse_and_rate_with_llm(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """
        Stage 2 API Call: Parse objects/conditions and rate construction difficulty.
        
        Args:
            problem_text: Cleaned problem text
            problem_id: Problem identifier
            
        Returns:
            Parsed problem data with difficulty rating
        """
        prompt = self._create_parsing_and_rating_prompt(problem_text)
        
        try:
            response = self.llm_api(prompt)
            response = response.strip()
            
            # Extract JSON
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
                "verification_conditions": parsed_data.get("verification_conditions", []),
                "category": parsed_data.get("category", "Unknown"),
                "difficulty": parsed_data.get("difficulty", 3)
            }
            
            # Validate and auto-correct
            result = self._validate_and_correct_parsed_data(result)
            
            return result
            
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to rule-based")
            result = self._parse_rule_based(problem_text, problem_id)
            result["category"] = "Unknown"
            result["difficulty"] = 3
            return result
    
    def _create_parsing_and_rating_prompt(self, problem_text: str) -> str:
        """Create prompt for Stage 2: Parsing and difficulty rating."""
        categories_str = "\n".join([f"- {cat}" for cat in self.PROBLEM_CATEGORIES])
        
        prompt = f"""Parse this geometry problem and extract construction requirements.

Problem: {problem_text}

**TASK 1: Extract geometric objects and conditions**

Return required_objects and verification_conditions as specified below.

**TASK 2: Classify into ONE category**:
{categories_str}

**TASK 3: Rate CONSTRUCTION difficulty (1-5)**

Rate how difficult it is to DRAW/CONSTRUCT the figure, NOT solve the problem.

1 = Very Easy: Simple shapes (single triangle, rectangle)
   - Example: "在△ABC中,∠C=90°"
   
2 = Easy: Basic constructions, few constraints
   - Example: "△ABC中,D是BC中点,AD⊥BC"
   
3 = Medium: Multiple objects, several relationships
   - Example: "AB∥CD,EF交AB于E,交CD于F,∠BEF=50°"
   
4 = Hard: Complex constructions, many objects, requires planning
   - Example: "四边形ABCD,E在AB上,F在CD上,EF∥AD,AG∥EF"
   
5 = Very Hard: Very complex, many interdependent constraints
   - Example: Multiple circles, tangent lines, concurrent lines, etc.

**Construction difficulty factors**:
- Number of points, lines, circles
- Dependent constructions (point depends on intersection, etc.)
- Precision requirements (specific angles, lengths)
- Need for auxiliary constructions

Return JSON:
{{
    "required_objects": {{
        "points": ["A", "B", "C"],
        "segments": [["A", "B"], ["B", "C"]],
        "lines": [],
        "circles": [{{"center": "O", "radius_point": "A"}}],
        "polygons": [["A", "B", "C"]]
    }},
    "verification_conditions": [
        // Basic geometric relationships
        {{"type": "parallel", "objects": [["A", "B"], ["C", "D"]]}},
        {{"type": "perpendicular", "objects": [["A", "B"], ["C", "D"]]}},
        {{"type": "collinear", "points": ["A", "B", "C"]}},
        {{"type": "not_collinear", "points": ["A", "B", "C"]}},
        {{"type": "concurrent", "lines": [["A", "B"], ["C", "D"], ["E", "F"]]}},
        
        // Angle conditions
        {{"type": "angle_value", "points": [["A", "B", "C"]], "value": 90}},
        {{"type": "angle_equality", "points": [["A", "B", "C"], ["D", "E", "F"]]}},
        {{"type": "angle_sum", "angles": [{{"points": ["A", "B", "C"]}}, {{"points": ["D", "E", "F"]}}], "value": 180}},
        {{"type": "angle_bisector", "line": ["E", "G"], "angle_points": ["B", "E", "F"]}},
        
        // Segment/Length conditions
        {{"type": "segment_equality", "segments": [["A", "B"], ["C", "D"]]}},
        {{"type": "distance_equals", "segment": ["A", "B"], "value": 5}},
        {{"type": "segment_length", "segment": ["A", "B"], "value": 10}},
        {{"type": "perimeter", "polygon": ["A", "B", "C"], "value": 20}},
        
        // Sum of segments conditions
        {{"type": "segments_sum_value", "segments": [["A", "B"], ["B", "C"]], "value": 15}},
        {{"type": "segments_sum_equals", "left_segments": [["A", "B"], ["B", "D"]], "right_segments": [["A", "C"]]}},
        {{"type": "ratio", "segments": [["A", "E"], ["E", "C"]], "ratio": [1, 2]}},
        
        // Point position conditions
        {{"type": "point_on_segment", "point": "D", "segment": ["A", "B"]}},
        {{"type": "point_on_line", "point": "D", "line": ["A", "B"]}},
        {{"type": "point_on_line_extension", "point": "E", "line_segment": ["A", "B"]}},
        {{"type": "point_on_segment_extension", "point": "E", "segment": ["A", "B"]}},
        {{"type": "midpoint_of", "point": "M", "segment": ["A", "B"]}},
        {{"type": "order_on_line", "points": ["A", "B", "C", "D"]}},
        {{"type": "same_side", "points": ["P", "Q"], "line": ["A", "B"]}},
        
        // Circle conditions
        {{"type": "point_on_circle", "point": "P", "circle_center": "O"}},
        {{"type": "concyclic", "points": ["A", "B", "C", "D"]}},
        {{"type": "tangent_line", "line": ["P", "T"], "circle_center": "O"}},
        {{"type": "tangent_at_point", "line": ["P", "T"], "circle_center": "O", "tangent_point": "T"}},
        {{"type": "diameter", "segment": ["A", "B"], "circle_center": "O"}},
        {{"type": "point_inside_circle", "point": "P", "circle_center": "O", "radius": 5}},
        
        // Polygon/Triangle conditions
        {{"type": "triangle_valid", "points": ["A", "B", "C"]}},
        {{"type": "isosceles_triangle", "points": ["A", "B", "C"], "equal_sides": [["A", "B"], ["A", "C"]]}},
        {{"type": "right_triangle", "points": ["A", "B", "C"]}},
        {{"type": "polygon_property", "polygon": ["A", "B", "C", "D"], "property": "parallelogram"}},
        {{"type": "polygon_type", "polygon": ["A", "B", "C", "D"], "value": "rhombus"}},
        {{"type": "square", "polygons": [["A", "B", "C", "D"]]}},
        {{"type": "regular_polygon", "polygon_points": ["A", "B", "C", "D", "E", "F"], "sides": 6}},
        
        // Special points
        {{"type": "intersection_point", "point": "P", "lines": [["A", "B"], ["C", "D"]]}},
        {{"type": "point_incenter", "point": "I", "triangle": ["A", "B", "C"]}},
        {{"type": "perpendicular_bisector", "line": ["M", "N"], "segment": ["A", "B"]}}
    ],
    "category": "Category Name",
    "difficulty": 3
}}

**SUPPORTED CONDITION TYPES**:
- Basic: parallel, perpendicular, collinear, not_collinear, concurrent
- Angles: angle_value, angle_equality, angle_sum, angle_bisector
- Segments: segment_equality, distance_equals, segment_length, perimeter, segments_sum_value, segments_sum_equals, ratio
- Points: point_on_segment, point_on_line, point_on_line_extension, midpoint_of, order_on_line, same_side
- Circles: point_on_circle, concyclic, tangent_line, tangent_at_point, diameter, point_inside_circle
- Polygons: triangle_valid, isosceles_triangle, right_triangle, polygon_property, polygon_type, square, regular_polygon
- Special: intersection_point, point_incenter, perpendicular_bisector

**LENGTH EXTRACTION RULES**:
1. "周长为30" or "周长=30" → {{"type": "perimeter", "polygon": [...], "value": 30}}
2. "AB=5" or "AB=5cm" → {{"type": "segment_length", "segment": ["A", "B"], "value": 5}}
3. "AB=BC" → {{"type": "segment_equality", "segments": [["A", "B"], ["B", "C"]]}}
4. "AB+BC=10" → {{"type": "segments_sum_value", "segments": [["A", "B"], ["B", "C"]], "value": 10}}
5. "AB+BD=AC" → {{"type": "segments_sum_equals", "left_segments": [["A", "B"], ["B", "D"]], "right_segments": [["A", "C"]]}}
6. "AE:EC=1:2" or "AE是EC的一半" → {{"type": "ratio", "segments": [["A", "E"], ["E", "C"]], "ratio": [1, 2]}}

**CRITICAL RULES**:
1. angle_value points MUST be nested: [["A", "B", "C"]] with B as vertex
2. parallel/perpendicular use "objects" field with 2D array
3. All points in conditions must be in required_objects.points
4. difficulty is INTEGER 1-5 based on CONSTRUCTION complexity
5. Use ONLY the supported condition types listed above

Only return JSON, no other text."""
        return prompt
    
    # Legacy method kept for compatibility
    def clean_problem_text(self, problem_text: str) -> str:
        """Legacy method - now uses _basic_clean_text internally."""
        return self._basic_clean_text(problem_text)
    
    def classify_problem(self, problem_text: str) -> str:
        """Classify problem - now integrated into Stage 2 parsing."""
        # This is now done in _parse_and_rate_with_llm
        return "Unknown"
    
    def rate_difficulty(self, problem_text: str) -> int:
        """Rate difficulty - now integrated into Stage 2 parsing."""
        # This is now done in _parse_and_rate_with_llm
        return 3
    
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
        
        # Check for length conditions (长度, =数字, cm, 周长)
        length_conditions = self._extract_length_conditions(problem_text, points)
        conditions.extend(length_conditions)
        
        # Check for perimeter conditions (周长)
        if '周长' in problem_text:
            perimeter_conditions = self._extract_perimeter_conditions(problem_text, points)
            conditions.extend(perimeter_conditions)
        
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
    
    def _extract_length_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract segment length conditions from text."""
        conditions = []
        
        # First, find all patterns that should NOT be treated as simple segment_length
        # These are segments that appear in sum or ratio patterns
        
        # Find segments in sum patterns: AB+BC=... 
        sum_segments = set()
        sum_pattern = r'([A-Z]{2})\s*\+\s*([A-Z]{2})[=等于]'
        for match in re.finditer(sum_pattern, text):
            sum_segments.add(match.group(1))
            sum_segments.add(match.group(2))
        
        # Find segments in ratio patterns: AE:EC=...
        ratio_segments = set()
        ratio_pattern = r'([A-Z]{2})\s*[:：]\s*([A-Z]{2})\s*[=等于]'
        for match in re.finditer(ratio_pattern, text):
            ratio_segments.add(match.group(1))
            ratio_segments.add(match.group(2))
        
        # Pattern: AB=5 or AB=5cm or AB=5厘米
        # Match segment with numeric value, but exclude those in sum/ratio patterns
        length_pattern = r'(?<!\+\s)([A-Z]{2})[=等于]\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米|CM)?(?![A-Z:：])'
        matches = re.findall(length_pattern, text)
        
        for match in matches:
            segment_str = match[0]
            # Skip if this segment is part of a sum pattern (the second segment after +)
            if segment_str in sum_segments or segment_str in ratio_segments:
                continue
            segment = list(segment_str)  # "AB" -> ["A", "B"]
            value = float(match[1])
            conditions.append({
                "type": "segment_length",
                "segment": segment,
                "value": value
            })
        
        # Pattern: AB=BC (segment equality without numeric value)
        equality_pattern = r'([A-Z]{2})[=等于]+([A-Z]{2})(?!\d)(?!\s*[:：])'
        matches = re.findall(equality_pattern, text)
        
        for match in matches:
            seg1 = list(match[0])
            seg2 = list(match[1])
            # Check it's not already captured as length condition
            if seg1[0] in points and seg1[1] in points and seg2[0] in points and seg2[1] in points:
                conditions.append({
                    "type": "segment_equality",
                    "segments": [seg1, seg2]
                })
        
        # Pattern: AB+BC=10 (sum of segments equals value)
        sum_value_pattern = r'([A-Z]{2})\s*\+\s*([A-Z]{2})[=等于]\s*(\d+(?:\.\d+)?)'
        matches = re.findall(sum_value_pattern, text)
        
        for match in matches:
            seg1 = list(match[0])
            seg2 = list(match[1])
            value = float(match[2])
            conditions.append({
                "type": "segments_sum_value",
                "segments": [seg1, seg2],
                "value": value
            })
        
        # Pattern: AB+BD=AC (sum equals another segment)
        sum_equals_pattern = r'([A-Z]{2})\s*\+\s*([A-Z]{2})[=等于]+([A-Z]{2})(?!\d)'
        matches = re.findall(sum_equals_pattern, text)
        
        for match in matches:
            seg1 = list(match[0])
            seg2 = list(match[1])
            target_seg = list(match[2])
            conditions.append({
                "type": "segments_sum_equals",
                "left_segments": [seg1, seg2],
                "right_segments": [target_seg]
            })
        
        # Pattern: AE:EC=1:2 or AE:EC=m:n (ratio)
        ratio_full_pattern = r'([A-Z]{2})\s*[:：]\s*([A-Z]{2})\s*[=等于]\s*(\d+)\s*[:：]\s*(\d+)'
        matches = re.findall(ratio_full_pattern, text)
        
        for match in matches:
            seg1 = list(match[0])
            seg2 = list(match[1])
            ratio1 = int(match[2])
            ratio2 = int(match[3])
            conditions.append({
                "type": "ratio",
                "segments": [seg1, seg2],
                "ratio": [ratio1, ratio2]
            })
        
        return conditions
    
    def _extract_perimeter_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract perimeter conditions from text."""
        conditions = []
        found_specific = False
        
        # Pattern: △ABC的周长为30 or 三角形ABC周长=30
        triangle_perimeter_pattern = r'[△三角形]\s*([A-Z]{3})[的]?周长[为=等于]\s*(\d+(?:\.\d+)?)'
        matches = re.findall(triangle_perimeter_pattern, text)
        
        for match in matches:
            polygon = list(match[0])  # "ABC" -> ["A", "B", "C"]
            value = float(match[1])
            conditions.append({
                "type": "perimeter",
                "polygon": polygon,
                "value": value
            })
            found_specific = True
        
        # Pattern: 四边形ABCD的周长为40
        quad_perimeter_pattern = r'四边形\s*([A-Z]{4})[的]?周长[为=等于]\s*(\d+(?:\.\d+)?)'
        quad_matches = re.findall(quad_perimeter_pattern, text)
        
        for match in quad_matches:
            polygon = list(match[0])  # "ABCD" -> ["A", "B", "C", "D"]
            value = float(match[1])
            conditions.append({
                "type": "perimeter",
                "polygon": polygon,
                "value": value
            })
            found_specific = True
        
        # Generic perimeter pattern without specific shape reference
        # Pattern: 周长为30cm or 周长=20
        # Only process if no specific patterns matched
        if not found_specific:
            generic_perimeter_pattern = r'周长[为=等于]\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米|CM)?'
            generic_matches = re.findall(generic_perimeter_pattern, text)
            
            # Try to infer the polygon from context
            # Look for △ABC or 三角形ABC or 四边形ABCD
            triangle_match = re.search(r'[△三角形]\s*([A-Z]{3})', text)
            quad_match = re.search(r'四边形\s*([A-Z]{4})', text)
            
            for value_str in generic_matches:
                value = float(value_str)
                if triangle_match:
                    polygon = list(triangle_match.group(1))
                    conditions.append({
                        "type": "perimeter",
                        "polygon": polygon,
                        "value": value
                    })
                elif quad_match:
                    polygon = list(quad_match.group(1))
                    conditions.append({
                        "type": "perimeter",
                        "polygon": polygon,
                        "value": value
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
    
    
    def parse_from_json(self, json_file: str, skip_ambiguous: bool = True, 
                       clean_text: bool = True) -> Optional[Dict[str, Any]]:
        """
        Parse problem from JSON file (like GeoQA3 format).
        
        Args:
            json_file: Path to JSON file
            skip_ambiguous: If True, skip problems with ambiguous references
            clean_text: Ignored (cleaning is now part of Stage 1)
            
        Returns:
            Parsed problem data, or None if skipped
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problem_text = data.get('subject', '')
        problem_id = str(data.get('id', 'unknown'))
        
        return self.parse_problem(problem_text, problem_id, skip_ambiguous)
    
    def save_to_benchmark_format(self, parsed_data: Dict[str, Any], output_file: str):
        """Save parsed data to benchmark format JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    
    def batch_parse_directory(self, input_dir: str, output_file: str, 
                             skip_ambiguous: bool = True, clean_text: bool = True,
                             file_pattern: str = "*.json") -> Dict[str, Any]:
        """
        Batch parse all JSON files in a directory and save to dataset.
        
        Args:
            input_dir: Directory containing JSON problem files
            output_file: Output file path for the dataset
            skip_ambiguous: If True, skip problems with ambiguous references
            clean_text: If True, clean the problem text before parsing
            file_pattern: Glob pattern for JSON files (default: "*.json")
            
        Returns:
            Dictionary with statistics about the parsing process
        """
        import glob
        
        json_files = glob.glob(os.path.join(input_dir, file_pattern))
        
        parsed_problems = []
        skipped_count = 0
        error_count = 0
        
        print(f"Found {len(json_files)} files to process in {input_dir}")
        print(f"Skip ambiguous: {skip_ambiguous}, Clean text: {clean_text}\n")
        
        for json_file in sorted(json_files):
            try:
                result = self.parse_from_json(json_file, skip_ambiguous, clean_text)
                
                if result is None:
                    skipped_count += 1
                else:
                    parsed_problems.append(result)
                    print(f"✓ Parsed: {json_file} (ID: {result.get('id', 'unknown')})")
                    if 'category' in result:
                        print(f"  Category: {result['category']}, Difficulty: {result.get('difficulty', 'N/A')}")
                
            except Exception as e:
                error_count += 1
                print(f"✗ Error parsing {json_file}: {e}")
        
        # Save the dataset
        dataset = {
            "metadata": {
                "total_files": len(json_files),
                "parsed": len(parsed_problems),
                "skipped": skipped_count,
                "errors": error_count,
                "skip_ambiguous": skip_ambiguous,
                "clean_text": clean_text
            },
            "problems": parsed_problems
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Dataset saved to: {output_file}")
        print(f"Total files: {len(json_files)}")
        print(f"Successfully parsed: {len(parsed_problems)}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        print('='*60)
        
        # Print category distribution if available
        if parsed_problems and 'category' in parsed_problems[0]:
            categories = {}
            difficulties = {}
            
            for problem in parsed_problems:
                cat = problem.get('category', 'Unknown')
                diff = problem.get('difficulty', 0)
                
                categories[cat] = categories.get(cat, 0) + 1
                difficulties[diff] = difficulties.get(diff, 0) + 1
            
            print("\nCategory Distribution:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count}")
            
            print("\nDifficulty Distribution:")
            for diff, count in sorted(difficulties.items()):
                print(f"  Level {diff}: {count}")
        
        return dataset["metadata"]


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
    print("="*70)
    print("Problem Parser - Two-Stage LLM Processing")
    print("Stage 1: Validate problem suitability + clean text")
    print("Stage 2: Extract objects/conditions + rate construction difficulty")
    print("="*70)
    
    # Setup parser
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        llm_function = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
        parser = ProblemParser(llm_api_function=llm_function)
        print("\n✓ Using OpenAI API for two-stage processing\n")
    else:
        print("\n⚠️  OPENAI_API_KEY not found, using rule-based parsing only\n")
        parser = ProblemParser()
    
    # Test with example problems - including some that should be rejected
    test_problems = [
        # Should PASS: Well-defined problem
        ("如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()", "test1"),
        
        # Should FAIL: Numbered angles (∠1, ∠2)
        ("如图所示,∠1=30°,∠2=45°,求∠3的大小", "test2"),
        
        # Should FAIL: Undefined point E (∠E without E's position)
        ("AB∥CD,∠E=40°,∠A=110°", "test3"),
        
        # Should FAIL: D's position not defined (∠BDC but where is D?)
        ("在△ABC中,∠C=90°,∠BDC=30°,AD=2BC", "test4"),
        
        # Should FAIL: Single-letter angle ∠D without context
        ("AB∥CD,∠D=26°,∠E=35°", "test5"),
        
        # Should PASS: Well-defined triangle
        ("在三角形ABC中,AB=BC,∠ABC=90°", "test6"),
        
        # Should PASS: Point position defined
        ("(3分)如图,在△ABC中,D在AB上,∠ACD=∠B,CD=4,那么AC·BC的值为()", "test7"),
    ]
    
    passed = 0
    failed = 0
    
    for problem_text, problem_id in test_problems:
        print(f"\n{'='*70}")
        print(f"Problem {problem_id}: {problem_text[:60]}...")
        print('='*70)
        
        result = parser.parse_problem(
            problem_text, 
            problem_id=problem_id,
            skip_ambiguous=True
        )
        
        if result is None:
            print("❌ Problem REJECTED (ambiguous/undefined references)")
            failed += 1
        else:
            print("✓ Problem ACCEPTED")
            print(f"  Cleaned: {result['cleaned_text'][:60]}...")
            print(f"  Category: {result.get('category', 'N/A')}")
            print(f"  Construction Difficulty: {result.get('difficulty', 'N/A')}/5")
            print(f"  Objects: {len(result.get('required_objects', {}).get('points', []))} points")
            print(f"  Conditions: {len(result.get('verification_conditions', []))} conditions")
            passed += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed} passed, {failed} rejected out of {len(test_problems)} problems")
    print("="*70)

