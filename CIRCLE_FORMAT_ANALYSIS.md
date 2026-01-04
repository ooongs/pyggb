# Circle Format Analysis Report

## Executive Summary

Analysis of `required_objects.circles` in geoqa3_dataset.json reveals:
- **Total circles**: 155 across 154 problems
- **Format distribution**:
  - ✓ `radius_point`: 151 (97.4%) - **Fully supported**
  - ✓ `radius_length`: 1 (0.6%) - **Partially supported**
  - ❌ `center` only: 2 (1.3%) - **Risky**
  - ❌ Wrong key `radius`: 1 (0.6%) - **Invalid**

## Supported Formats

### 1. radius_point Format (RECOMMENDED)
```json
{
  "center": "O",
  "radius_point": "A"
}
```
- **Usage**: 151 circles (97.4%)
- **Status**: ✓ Fully supported
- **Validator behavior**: Finds circle by center only, does NOT validate radius_point
- **Issues**: None

### 2. radius_length Format
```json
{
  "center": "O",
  "radius_length": 10
}
```
- **Usage**: 1 circle (Problem 609)
- **Status**: ✓ Partially supported
- **Validator behavior**: Finds circle by center only, does NOT validate radius_length
- **Issues**: Radius value is ignored during validation

## Invalid/Risky Formats

### 3. Center Only Format (RISKY)
```json
{
  "center": "O"
}
```
- **Usage**: 2 circles (Problems 288, 963)
- **Status**: ⚠️ Risky - works only if one circle with that center
- **Issues**:
  - Missing radius specification
  - Will accept any circle with matching center
  - May cause silent failures

### 4. Wrong Key 'radius' (INVALID)
```json
{
  "center": "O",
  "radius": 6
}
```
- **Usage**: 1 circle (Problem 595)
- **Status**: ❌ Invalid format
- **Issues**: Should use `radius_length` instead of `radius`

## Validator Implementation Analysis

### Current Behavior (dsl_validator.py:496-506)
```python
for circle_def in required_objects.circles:
    center = circle_def.get("center")
    if center and center in element_dict:
        circle_label = self._find_circle_with_center(center)
        if circle_label:
            found["circles"].append(circle_def)
        else:
            missing["circles"].append(circle_def)
```

**Key Finding**: The validator only checks for center existence, NOT radius!

### _find_circle_with_center (dsl_validator.py:678-695)
```python
def _find_circle_with_center(self, center: str) -> Optional[str]:
    """Find a circle with given center point."""
    # Returns the FIRST circle found with matching center
    # Does NOT check radius_point or radius_length
```

## Problems Requiring Fixes

| Problem ID | Current Format | Issue | Recommended Fix |
|------------|----------------|-------|-----------------|
| **595** | `{'center': 'O', 'radius': 6}` | Wrong key `radius` | Change to `radius_length: 6` |
| **288** | `{'center': 'O'}` | No radius spec | Add `radius_point` or `radius_length` |
| **963** | `{'center': 'O'}` | No radius spec | Add `radius_point` or `radius_length` |

## Concentric Circles Support

**Status**: ✓ Validator has `_find_all_circles_with_center()` method
**Dataset**: No concentric circles found (0 instances)

The validator includes support for multiple circles with the same center, but this feature is not currently used in the dataset.

## Recommendations

### 1. Fix Invalid Formats (HIGH PRIORITY)
```bash
# Problem 595: Change 'radius' to 'radius_length'
# Problems 288, 963: Add radius specification
```

### 2. Enhance Validator (MEDIUM PRIORITY)
The validator should verify radius matches when specified:
- For `radius_point`: Check that the point is on the circle
- For `radius_length`: Check that the circle radius equals the specified value

### 3. Dataset Standards (LOW PRIORITY)
Establish clear guidelines:
- **Prefer**: `radius_point` format when a radius point exists
- **Use**: `radius_length` only when radius is a specific numeric value
- **Never**: Omit radius specification
- **Never**: Use `radius` (always use `radius_length`)

## Code Examples

### Fix for Problem 595
```python
# Before (INVALID)
"circles": [{"center": "O", "radius": 6}]

# After (VALID)
"circles": [{"center": "O", "radius_length": 6}]
```

### Fix for Problems 288, 963
```python
# Before (RISKY)
"circles": [{"center": "O"}]

# After (if radius point A exists)
"circles": [{"center": "O", "radius_point": "A"}]

# After (if only numeric radius known)
"circles": [{"center": "O", "radius_length": 5}]
```

## Validation Test Results

**Both formats are accepted by the validator** (center matching only):
- ✓ `{"center": "O", "radius_point": "A"}` - Works
- ✓ `{"center": "O", "radius_length": 10}` - Works
- ⚠️ `{"center": "O"}` - Works but risky
- ❌ `{"center": "O", "radius": 6}` - Invalid key, should fail

**Note**: The validator does NOT validate that the radius matches the specified value. It only checks that a circle with the given center exists.
