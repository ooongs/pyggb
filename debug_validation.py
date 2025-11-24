#!/usr/bin/env python3
"""Debug validation issue"""

import tempfile
from dsl_validator import DSLValidator
from benchmark_dataset import BenchmarkDataset

# Load the dataset
dataset = BenchmarkDataset("benchmark_geoqa3.json")
problem = dataset.get_problem("0")

print("Problem:", problem.subject)
print("\nRequired Objects:")
print("  Points:", problem.required_objects.points)
print("  Segments:", problem.required_objects.segments)
print("  Lines:", problem.required_objects.lines)
print("  Polygons:", problem.required_objects.polygons)
print("\nVerification Conditions:", len(problem.verification_conditions))
for i, cond in enumerate(problem.verification_conditions):
    print(f"  {i+1}. {cond.type}: {cond.data}")

# The DSL from iteration 2
dsl_code = """# Place A at (0,0)
point : 0 0 -> A
# Place B at (200,0)
point : 200 0 -> B
# Place C at (47,265) to satisfy the triangle's angles (precomputed)
point : 47 265 -> C

# Draw triangle
polygon : A B C -> triangle c a b

# Place D on AB (not at endpoints)
point : 80 0 -> D

# Draw BC
line : B C -> line_BC

# Draw AC
line : A C -> line_AC

# Construct a line through D parallel to BC
# To do this, create a point E_dir by moving from D by the direction vector of BC
# BC vector: (47-200, 265-0) = (-153, 265)
point : -73 265 -> E_dir
line : D E_dir -> line_DE_parallel

# Intersect this line with AC to get E
intersect : line_DE_parallel line_AC -> E

# Draw DE
segment : D E -> seg_DE

# Draw CE
segment : C E -> seg_CE

# Construct angle CED (angle at E between C and D)
angle : C E D -> angle_CED

# End with prove statement
equality : A A -> expr0
prove : expr0 -> result
"""

# Save to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(dsl_code)
    dsl_file = f.name

print("\nValidating DSL...")
print("="*70)

# Enable debug mode
import sys
sys._debug_validation = True

# Validate
validator = DSLValidator()
result = validator.validate(dsl_file, problem, max_attempts=10)

print("\n" + "="*70)
print("VALIDATION RESULT:")
print("="*70)
print(f"Success: {result.success}")
print(f"Object Score: {result.object_score:.1%}")
print(f"Condition Score: {result.condition_score:.1%}")
print(f"Total Score: {result.total_score:.1%}")

if result.error_message:
    print(f"\nError: {result.error_message}")

print(f"\nMissing Objects:")
for obj_type, objs in result.missing_objects.items():
    if objs:
        print(f"  {obj_type}: {objs}")

print(f"\nFailed Conditions: {len(result.failed_conditions)}")
for fc in result.failed_conditions:
    print(f"  - {fc.get('type')}: {fc.get('validation_message', 'No message')}")

if result.details:
    print(f"\nDetailed Object Check:")
    obj_details = result.details.get('object_details', {})
    print(f"  Total Required: {obj_details.get('total_required')}")
    print(f"  Total Found: {obj_details.get('total_found')}")
    
    print(f"\nDetailed Condition Check:")
    cond_details = result.details.get('condition_details', [])
    for i, detail in enumerate(cond_details):
        cond = detail.get('condition', {})
        passed = detail.get('passed', False)
        message = detail.get('message', 'No message')
        print(f"  {i+1}. {cond.get('type')}: {'✓' if passed else '✗'} - {message}")

# Cleanup
import os
os.remove(dsl_file)

