# GeoBuildBench - Geometry Construction from Text

Python tool for generating geometric constructions from problem descriptions using multimodal LLMs.

## Features

- **Geometry Problem Parsing**: Extract geometric objects and verification conditions from text
- **DSL-based Construction**: Domain-specific language for geometric constructions
- **Mathematical Expressions**: Arithmetic and trigonometric expressions in DSL (e.g., `100*cos(45°)`, `50+30*sin(60°)`)
- **Multimodal LLM Agent**: ReAct agent using vision-language models
- **Benchmark System**: Evaluate LLM performance on geometry tasks
- **Visualization**: Interactive matplotlib-based viewer

## Installation

```bash
# Using conda (recommended)
conda activate your_env
pip install -r requirements.txt

# Or using pip with virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
```

## Quick Start

### Run Benchmark

```bash
# Single problem
python run_agent_benchmark.py --problem-id 0 --model gpt-4o --verbose

# Batch mode
python run_agent_benchmark.py --batch --model gpt-4o --limit 10
```

### Interactive Viewer

```bash
python preview.py
```

**Controls:**

- **Arrow keys**: Navigate between constructions
- **Space**: Regenerate current construction
- **Escape**: Exit

## Project Structure

```
pyggb/
├── src/                          # Core source code
│   ├── core/                     # Geometry primitives and commands
│   ├── dsl/                      # DSL executor and validator
│   ├── agent/                    # ReAct agent implementation
│   ├── benchmark/                # Benchmark dataset handling
│   ├── interfaces/               # LLM interfaces
│   ├── parser/                   # Problem parsing
│   ├── ggb/                      # GeoGebra integration
│   └── utils.py                  # Path utilities
├── scripts/                      # Utility scripts
│   ├── create_dataset.py         # Dataset creation
│   ├── analyze_benchmark_results.py  # Result analysis
│   └── ...
├── prompts/                      # Agent prompts
├── data/                 # Benchmark datasets
├── run_agent_benchmark.py        # Main execution script
├── preview.py                    # Visualization tool
└── *.sh                          # Shell scripts for benchmarking
```

## Shell Scripts

| Script                         | Description                |
| ------------------------------ | -------------------------- |
| `run_multi_model_benchmark.sh` | Compare multiple models    |
| `run_vision_comparison.sh`     | Vision model benchmarks    |
| `run_parallel_dataset.sh`      | Parallel dataset creation  |
| `run_parallel_simple.sh`       | Simple parallel processing |

## Documentation

| Document                                     | Description                                 |
| -------------------------------------------- | ------------------------------------------- |
| [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) | Parser, validation, and parallel processing |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)     | Running and analyzing benchmarks            |
| [CHANGELOG.md](CHANGELOG.md)                 | Version history and changes                 |

## Core Modules

### Geometry Types (`src/core/geo_types.py`)

Geometric primitives with matplotlib rendering:

- Point, Line, Segment
- Circle, Arc
- Polygon, Triangle

### Commands (`src/core/commands.py`)

Construction commands:

- `line_pp`: Line through two points
- `intersect_ll`: Line-line intersection
- `circle_cr`: Circle with center and radius
- And many more...

### DSL Executor (`src/dsl/dsl_executor.py`)

Executes DSL code and renders constructions:

```python
from src.dsl.dsl_executor import DSLExecutor

executor = DSLExecutor()

# Simple construction
executor.execute("""
point :  -> A
point :  -> B
segment : A B -> seg_AB
""")

# With mathematical expressions 
executor.execute("""
point : 0 0 -> O
point : 100*cos(0°) 100*sin(0°) -> A
point : 100*cos(120°) 100*sin(120°) -> B
point : 100*cos(240°) 100*sin(240°) -> C
polygon : A B C -> triangle AB BC CA
""")
```

### ReAct Agent (`src/agent/react_agent.py`)

Multimodal agent for geometry problem-solving:

```python
from src.agent.react_agent import ReActAgent
from src.benchmark.benchmark_dataset import BenchmarkDataset

dataset = BenchmarkDataset("data/geoqa3_dataset.json")
problem = dataset.get_problem("123")

agent = ReActAgent(model="gpt-4o")
result = agent.solve(problem)
```

## License

MIT License
