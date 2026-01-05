#!/bin/bash
#
# Multi-Model Benchmark Runner
# Run benchmark tests on multiple vLLM models for comparison
#
# Usage:
#   ./run_multi_model_benchmark.sh [options]
#
# Examples:
#   ./run_multi_model_benchmark.sh --limit 10 --models "gpt-4o,claude-3-5-sonnet-20241022"
#   ./run_multi_model_benchmark.sh --full --api-base http://localhost:8000/v1
#

set -e

# Default configuration
DATASET="data/geoqa3_dataset.json"
MAX_ITER=5
LIMIT=""
START_IDX=0
OUTPUT_DIR="benchmark_results"
VERBOSE=""
API_BASE=""

# Default models list (customize as needed)
DEFAULT_MODELS=(
    # "gpt-5.1"
    # "gpt-4.1"
    # "gpt-5.1"
    # "claude-sonnet-4-5"
    # Add vLLM models here
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # "gemini-2.5-pro"
    "gemini-3-flash-preview"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║              Multi-Model Benchmark Runner                        ║"
    echo "║              Geometry Problem Solving Agent                      ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --models MODELS     Comma-separated list of models to test"
    echo "  -d, --dataset FILE      Dataset file path (default: $DATASET)"
    echo "  -i, --max-iter N        Maximum iterations per problem (default: $MAX_ITER)"
    echo "  -l, --limit N           Limit number of problems (default: all)"
    echo "  -s, --start-idx N       Starting index (default: 0)"
    echo "  -o, --output-dir DIR    Output directory (default: $OUTPUT_DIR)"
    echo "  -a, --api-base URL      API base URL for vLLM (default: none)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  --full                  Run on full dataset (no limit)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default OpenAI models, limit 10 problems"
    echo "  $0 --limit 10"
    echo ""
    echo "  # Run with specific models"
    echo "  $0 --models 'gpt-4o,gpt-4o-mini' --limit 20"
    echo ""
    echo "  # Run with vLLM models"
    echo "  $0 --api-base http://localhost:8000/v1 --models 'Qwen/Qwen2.5-VL-7B-Instruct'"
    echo ""
    echo "  # Run full benchmark"
    echo "  $0 --full --models 'gpt-4o'"
}

# Parse arguments
MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -i|--max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        -s|--start-idx)
            START_IDX="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -a|--api-base)
            API_BASE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --full)
            LIMIT=""
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Use default models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

print_banner

echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET"
echo "  Max iterations: $MAX_ITER"
echo "  Start index: $START_IDX"
echo "  Limit: ${LIMIT:-'None (full dataset)'}"
echo "  Output directory: $RUN_DIR"
echo "  API base: ${API_BASE:-'Default'}"
echo "  Models to test: ${MODELS[*]}"
echo ""

# Log file
LOG_FILE="$RUN_DIR/benchmark_run.log"
echo "Benchmark run started at $(date)" > "$LOG_FILE"
echo "Models: ${MODELS[*]}" >> "$LOG_FILE"

# Run benchmark for each model
RESULT_FILES=()
SUCCESS_COUNT=0
FAIL_COUNT=0

for MODEL in "${MODELS[@]}"; do
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Testing model: $MODEL${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    
    # Sanitize model name for filename
    MODEL_SAFE=$(echo "$MODEL" | tr '/:' '__')
    OUTPUT_FILE="$RUN_DIR/results_${MODEL_SAFE}.json"
    
    # Build command
    CMD="python run_agent_benchmark.py --batch"
    CMD="$CMD --model '$MODEL'"
    CMD="$CMD --dataset '$DATASET'"
    CMD="$CMD --max-iter $MAX_ITER"
    CMD="$CMD --start-idx $START_IDX"
    CMD="$CMD --output '$OUTPUT_FILE'"
    CMD="$CMD $LIMIT"
    CMD="$CMD $VERBOSE"
    CMD="$CMD --no-vision"
    
    # Set API base if provided
    if [ -n "$API_BASE" ]; then
        export OPENAI_API_BASE="$API_BASE"
    fi
    
    echo "Command: $CMD"
    echo ""
    
    # Run and capture result
    echo "Starting benchmark for $MODEL at $(date)" >> "$LOG_FILE"
    
    if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Completed: $MODEL${NC}"
        echo "Completed: $MODEL at $(date)" >> "$LOG_FILE"
        RESULT_FILES+=("$OUTPUT_FILE")
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ Failed: $MODEL${NC}"
        echo "Failed: $MODEL at $(date)" >> "$LOG_FILE"
        ((FAIL_COUNT++))
    fi
    
    echo ""
done

# Generate comparison report
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Generating Comparison Report${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    REPORT_FILE="$RUN_DIR/comparison_report.txt"
    CSV_FILE="$RUN_DIR/comparison_summary.csv"
    JSON_FILE="$RUN_DIR/comparison_report.json"
    
    # Generate text report
    python scripts/analyze_benchmark_results.py "${RESULT_FILES[@]}" --format text --output "$REPORT_FILE" --per-problem
    
    # Generate CSV for spreadsheet
    python scripts/analyze_benchmark_results.py "${RESULT_FILES[@]}" --format csv --output "$CSV_FILE"
    
    # Generate JSON for programmatic access
    python scripts/analyze_benchmark_results.py "${RESULT_FILES[@]}" --format json --output "$JSON_FILE"
    
    echo -e "${GREEN}Reports generated:${NC}"
    echo "  Text report: $REPORT_FILE"
    echo "  CSV summary: $CSV_FILE"
    echo "  JSON report: $JSON_FILE"
    
    echo ""
    echo -e "${YELLOW}Summary:${NC}"
    cat "$REPORT_FILE"
fi

# Final summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Benchmark Run Complete${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Models tested: ${#MODELS[@]}"
echo -e "  ${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "  ${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo -e "  Results saved to: $RUN_DIR"
echo ""

echo "Benchmark completed at $(date)" >> "$LOG_FILE"


