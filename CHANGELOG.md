# Changelog

All notable changes to this project are documented in this file.

---

## [Unreleased]

### Polygon Command Removal (December 12, 2025)

Removed polygon command usage to enforce explicit segment creation.

#### Changes

**DSL Policy Update**
- Polygon command is now forbidden in agent-generated DSL
- All polygons (triangles, quadrilaterals, etc.) must be created using explicit segments
- Updated all documentation and prompts to reflect this change

**Benefits**
- More explicit and clear constructions
- Better control over geometric object creation
- Easier validation of required segments
- Clearer visualization (only points show labels)

**Documentation Updates**
- `prompts/system_prompt.txt`: Added rule #3 forbidding polygon command
- `prompts/dsl_guidelines.txt`: Updated section 6 with explicit segment examples
- `prompts/examples.txt`: All examples now use explicit segments
- `prompts/hints/error_polygon_forbidden.txt`: New error hint for polygon usage

**Example Change**
```
# Before (forbidden):
polygon : A B C -> triangle AB BC CA

# After (required):
segment : A B -> AB
segment : B C -> BC
segment : C A -> CA
```

**Rendering Changes**
- Only Point objects show labels in visualizations
- Segments, lines, circles are drawn but without labels
- Angles are no longer rendered (removed from drawable objects)

---

### Mathematical Expression Support (December 12, 2025)

Added powerful mathematical expression evaluation in DSL for more intuitive geometry construction.

#### New Features

**Expression Evaluation**
- Arithmetic operations: `+`, `-`, `*`, `/`, `()`
- Trigonometric functions: `cos()`, `sin()`, `tan()`
- Combined expressions: `100*cos(45°)`, `50+30*sin(60°)`
- Support for both degree (`°`, `deg`) and radian (`rad`, `r`) units

**Usage Examples**
```
# Polar coordinates for points
point : 100*cos(0°) 100*sin(0°) -> A
point : 100*cos(120°) 100*sin(120°) -> B
point : 100*cos(240°) 100*sin(240°) -> C

# Arithmetic expressions
point : 50+30 100-20 -> P
circle : O 100*sin(60°) -> c

# Complex expressions
point : 50+30*cos(45°) 50+30*sin(45°) -> Q
```

**Priority Rules**
1. Existing objects are referenced first (prevents ambiguity)
2. Expressions with operators are evaluated
3. Simple numbers are treated as literals
4. Clear error messages for invalid expressions

**Documentation Updates**
- Updated `prompts/dsl_guidelines.txt` with comprehensive expression syntax
- Added examples in `prompts/examples.txt` for regular polygons and polar coordinates
- Added trigonometric value reference table

#### Technical Details

**Implementation** (`src/core/random_constr.py`)
- New function `evaluate_math_expression()`: Safe expression evaluation with restricted namespace
- Updated `parse_command()`: Prioritizes existing objects over expression evaluation
- Enhanced `parse_trig_function()`: Handles degree/radian units with scientific notation

**Safety Features**
- Sandboxed evaluation using restricted `eval()` with no builtins
- Only allows: numbers, operators (+, -, *, /, ()), and trig functions
- Supports scientific notation (e.g., `1e-17`)

---

### Repository Reorganization (December 10, 2025)

Major restructuring of the repository for improved maintainability and clarity.

#### Directory Structure Changes

**New Structure:**

```
pyggb/
├── src/                          # Core source code
│   ├── core/                     # Core geometry modules
│   │   ├── geo_types.py
│   │   ├── commands.py
│   │   └── random_constr.py
│   ├── dsl/                      # DSL execution and validation
│   │   ├── dsl_executor.py
│   │   └── dsl_validator.py
│   ├── agent/                    # Agent implementation
│   │   ├── react_agent.py
│   │   ├── agent_memory.py
│   │   └── agent_logger.py
│   ├── benchmark/                # Benchmark dataset handling
│   │   └── benchmark_dataset.py
│   ├── interfaces/               # LLM interfaces
│   │   └── multimodal_interface.py
│   ├── parser/                   # Problem parsing
│   │   └── problem_parser.py
│   ├── ggb/                      # GeoGebra integration
│   │   ├── ggb_expr.py
│   │   └── read_ggb.py
│   └── utils.py                  # Path utilities
├── scripts/                      # Utility scripts
│   ├── analyze_benchmark_results.py
│   ├── create_dataset.py
│   ├── convert_geoqa_to_benchmark.py
│   ├── regenerate_dataset.py
│   ├── debug_validation.py
│   └── example_batch_process.py
├── prompts/                      # Agent prompts
├── examples/                     # Example files
├── data/                 # Benchmark datasets
├── *.sh                          # Shell scripts (root)
├── run_agent_benchmark.py        # Main execution script
└── preview.py                    # Visualization tool
```

#### Key Changes

1. **Core modules moved to `src/`**: All Python source code organized into subpackages
2. **Scripts moved to `scripts/`**: Dataset generation and analysis utilities
3. **Path handling improved**: All hardcoded paths replaced with `src/utils.py` functions
4. **Import structure updated**: Uses `src.subpackage.module` pattern
5. **Shell scripts updated**: Reference new script locations

#### Migration Notes

- Root execution scripts (`run_agent_benchmark.py`, `preview.py`) remain in root
- Shell scripts work exactly as before
- All functionality preserved - no logic changes

---

### Validation System Improvements

#### New Condition Handlers

Added 6 new condition types to `dsl_validator.py`:

| Condition            | Description                              |
| -------------------- | ---------------------------------------- |
| `point_on_segment`   | Check if point lies on segment           |
| `midpoint_of`        | Verify point is midpoint of segment      |
| `distance_equals`    | Check if distance matches expected value |
| `triangle_valid`     | Verify valid non-degenerate triangle     |
| `point_between`      | Check if point is between two others     |
| `concentric_circles` | Verify circles share same center         |

#### Hybrid Validation Mode

Implemented flexible validation that:

- **Explicitly validates**: Polygons, circles, labeled points
- **Implicitly infers**: Segments and lines from existing points

#### Enhanced Feedback

Validation now provides:

- Found objects listing
- Passed conditions with details
- Failed conditions with specific reasons
- Suggested DSL commands for fixes

---

### Bug Fix: Zero Validation Scores

#### Problem

Users saw contradictory messages:

```
Object Score: 0.0%
Condition Score: 0.0%
✅ All objects and conditions satisfied!
```

#### Root Cause

1. DSL syntax errors caused validation to fail silently
2. Logger didn't check score values before showing success message

#### Fixes Applied

1. **Logger updated**: Only shows success message when `total_score > 0`
2. **Validator enhanced**: Detects 0-score scenarios and adds explanatory messages
3. **Failed conditions**: Now include validation message explaining why they failed

---

### Problem Parser V2

#### Two-Stage Processing

| Stage   | Purpose                                           |
| ------- | ------------------------------------------------- |
| Stage 1 | Problem validation and text cleaning              |
| Stage 2 | Object/condition extraction and difficulty rating |

#### Enhanced Filtering

Now detects and skips:

- Undefined points (e.g., `∠E=40°` where E is not defined)
- Unlocated points (e.g., `∠BDC=30°` where D's position is undefined)
- Single-letter angles (e.g., `∠D=26°`)
- Numbered angles (e.g., `∠1=30°`)

#### Difficulty Rating Change

Difficulty now rates **construction complexity** (how hard to draw), not problem-solving difficulty.

| Level | Construction Complexity                       |
| ----- | --------------------------------------------- |
| 1     | Single triangle, rectangle                    |
| 2     | Basic constructions, few conditions           |
| 3     | Multiple objects, several relationships       |
| 4     | Complex constructions, many objects           |
| 5     | Multiple circles, tangents, concurrent points |

---

### Parallel Processing Support

#### New Features

1. **Incremental saving**: Each problem saved immediately to JSONL
2. **Range processing**: `--start` and `--end` flags for specific ranges
3. **Parallel execution**: Multiple workers can process simultaneously
4. **Resume capability**: Automatically skips already-processed IDs
5. **Merge function**: Combines JSONL files into final JSON dataset

#### Usage

```bash
# Parallel processing script
./run_parallel_dataset.sh 4 5000 1250  # 4 workers, 5000 files, 1250 per batch

# Manual parallel in multiple terminals
python scripts/create_dataset.py --start 0 --end 1250
python scripts/create_dataset.py --start 1250 --end 2500

# Merge results
python scripts/create_dataset.py --merge
```

---

### Parser Feature Additions

#### New Methods

| Method                       | Description                |
| ---------------------------- | -------------------------- |
| `has_ambiguous_references()` | Detect numbered angles     |
| `clean_problem_text()`       | Remove figure references   |
| `classify_problem()`         | 10-category classification |
| `rate_difficulty()`          | 1-5 difficulty rating      |
| `batch_parse_directory()`    | Batch processing           |

#### Text Cleaning

Automatically removes:

- Figure references: "如图所示", "如图", "图中"
- Question patterns: "则 ∠AOB 的大小是()", "求...()"

---

### Migration from Gtk/Cairo to Matplotlib

The codebase was modernized from Gtk 3.0/Cairo to Matplotlib.

#### Benefits

- Cross-platform compatibility (macOS, Linux, Windows)
- Easy installation with pip
- Rich plotting capabilities
- Better documentation and community support

#### Changes

- All geometric shapes use `matplotlib.axes.Axes` for rendering
- Interactive viewer uses matplotlib's event system
- Simpler installation process (no system packages needed)

---

## Documentation

### Consolidated Documentation

The following documentation files are available:

| File                   | Description                                 |
| ---------------------- | ------------------------------------------- |
| `README.md`            | Project overview and installation           |
| `DEVELOPMENT_GUIDE.md` | Parser, parallel processing, and validation |
| `BENCHMARK_GUIDE.md`   | Running and analyzing benchmarks            |
| `CHANGELOG.md`         | This file - all changes and updates         |

### Removed Files

The following files were consolidated into the above documents:

- `README_PARSER_UPDATES.md`
- `README_PARALLEL_PROCESSING.md`
- `README_BENCHMARK.md`
- `QUICK_START_PARSER.md`
- `PARSER_V2_CHANGES.md`
- `BUGFIX_ZERO_SCORES.md`
- `CHANGES_SUMMARY.md`
- `REORGANIZATION_COMPLETE.md`
- `REORGANIZATION_SUMMARY.md`
- `PROBLEM_PARSER_GUIDE.md`
- `PARALLEL_PROCESSING_GUIDE.md`
- `VALIDATION_DEBUGGING_GUIDE.md`
- `VALIDATION_IMPROVEMENTS.md`
- `VERIFICATION_CHECKLIST.md`


















