#!/bin/bash
#
# Vision Comparison Benchmark Runner
# Compare model performance with and without vision (rendered images)
#
# Usage:
#   ./run_vision_comparison.sh [options]
#
# This script runs the same benchmark twice:
#   1. With vision enabled (images sent to LLM)
#   2. With vision disabled (no images sent)
# Then generates a comparison report.
#

set -e

# Default configuration
DATASET="ground_truth/geoqa3_dataset.json"
MAX_ITER=5
LIMIT=""
START_IDX=0
OUTPUT_DIR="benchmark_results"
VERBOSE=""
MODEL="gpt-4o"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║              Vision Comparison Benchmark                         ║"
    echo "║         Compare Agent Performance With/Without Vision            ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Model to test (default: $MODEL)"
    echo "  -d, --dataset FILE      Dataset file path (default: $DATASET)"
    echo "  -i, --max-iter N        Maximum iterations per problem (default: $MAX_ITER)"
    echo "  -l, --limit N           Limit number of problems (default: all)"
    echo "  -s, --start-idx N       Starting index (default: 0)"
    echo "  -o, --output-dir DIR    Output directory (default: $OUTPUT_DIR)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Compare with 10 problems"
    echo "  $0 --model gpt-4o --limit 10"
    echo ""
    echo "  # Compare with vLLM model"
    echo "  $0 --model 'Qwen/Qwen2.5-VL-7B-Instruct' --limit 20"
    echo ""
    echo "  # Full comparison"
    echo "  $0 --model gpt-4o"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
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
        -v|--verbose)
            VERBOSE="--verbose"
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

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/vision_comparison_$TIMESTAMP"
mkdir -p "$RUN_DIR"

# Sanitize model name for filename
MODEL_SAFE=$(echo "$MODEL" | tr '/:' '__')

print_banner

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Max iterations: $MAX_ITER"
echo "  Start index: $START_IDX"
echo "  Limit: ${LIMIT:-'None (full dataset)'}"
echo "  Output directory: $RUN_DIR"
echo ""

# Log file
LOG_FILE="$RUN_DIR/comparison_run.log"
echo "Vision comparison started at $(date)" > "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"

# ============================================
# Run 1: WITH Vision
# ============================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}📷 Phase 1: Running WITH Vision${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

OUTPUT_VISION="$RUN_DIR/results_${MODEL_SAFE}_with_vision.json"

CMD_VISION="python run_agent_benchmark.py --batch"
CMD_VISION="$CMD_VISION --model '$MODEL'"
CMD_VISION="$CMD_VISION --dataset '$DATASET'"
CMD_VISION="$CMD_VISION --max-iter $MAX_ITER"
CMD_VISION="$CMD_VISION --start-idx $START_IDX"
CMD_VISION="$CMD_VISION --output '$OUTPUT_VISION'"
CMD_VISION="$CMD_VISION $LIMIT"
CMD_VISION="$CMD_VISION $VERBOSE"

echo "Command: $CMD_VISION"
echo ""

echo "With vision run started at $(date)" >> "$LOG_FILE"

if eval $CMD_VISION 2>&1 | tee -a "$LOG_FILE"; then
    echo -e "${GREEN}✓ Completed: WITH Vision${NC}"
    echo "With vision completed at $(date)" >> "$LOG_FILE"
else
    echo -e "${RED}✗ Failed: WITH Vision${NC}"
    echo "With vision failed at $(date)" >> "$LOG_FILE"
fi

# ============================================
# Run 2: WITHOUT Vision
# ============================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🔇 Phase 2: Running WITHOUT Vision${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

OUTPUT_NO_VISION="$RUN_DIR/results_${MODEL_SAFE}_no_vision.json"

CMD_NO_VISION="python run_agent_benchmark.py --batch"
CMD_NO_VISION="$CMD_NO_VISION --model '$MODEL'"
CMD_NO_VISION="$CMD_NO_VISION --dataset '$DATASET'"
CMD_NO_VISION="$CMD_NO_VISION --max-iter $MAX_ITER"
CMD_NO_VISION="$CMD_NO_VISION --start-idx $START_IDX"
CMD_NO_VISION="$CMD_NO_VISION --output '$OUTPUT_NO_VISION'"
CMD_NO_VISION="$CMD_NO_VISION $LIMIT"
CMD_NO_VISION="$CMD_NO_VISION $VERBOSE"
CMD_NO_VISION="$CMD_NO_VISION --no-vision"

echo "Command: $CMD_NO_VISION"
echo ""

echo "No vision run started at $(date)" >> "$LOG_FILE"

if eval $CMD_NO_VISION 2>&1 | tee -a "$LOG_FILE"; then
    echo -e "${GREEN}✓ Completed: WITHOUT Vision${NC}"
    echo "No vision completed at $(date)" >> "$LOG_FILE"
else
    echo -e "${RED}✗ Failed: WITHOUT Vision${NC}"
    echo "No vision failed at $(date)" >> "$LOG_FILE"
fi

# ============================================
# Generate Comparison Report
# ============================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 Generating Comparison Report${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

REPORT_FILE="$RUN_DIR/vision_comparison_report.txt"
CSV_FILE="$RUN_DIR/vision_comparison.csv"
JSON_FILE="$RUN_DIR/vision_comparison.json"

# Check if both result files exist
if [ -f "$OUTPUT_VISION" ] && [ -f "$OUTPUT_NO_VISION" ]; then
    # Generate comparison reports
    python analyze_benchmark_results.py "$OUTPUT_VISION" "$OUTPUT_NO_VISION" \
        --format text --output "$REPORT_FILE" --per-problem
    
    python analyze_benchmark_results.py "$OUTPUT_VISION" "$OUTPUT_NO_VISION" \
        --format csv --output "$CSV_FILE"
    
    python analyze_benchmark_results.py "$OUTPUT_VISION" "$OUTPUT_NO_VISION" \
        --format json --output "$JSON_FILE"
    
    echo -e "${GREEN}Reports generated:${NC}"
    echo "  Text report: $REPORT_FILE"
    echo "  CSV summary: $CSV_FILE"
    echo "  JSON report: $JSON_FILE"
    
    # Print summary
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}📈 VISION COMPARISON SUMMARY${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
    
    # Extract key metrics using Python
    python3 << EOF
import json

with open("$OUTPUT_VISION", 'r') as f:
    with_vision = json.load(f)

with open("$OUTPUT_NO_VISION", 'r') as f:
    no_vision = json.load(f)

v_metrics = with_vision.get("metrics", {})
nv_metrics = no_vision.get("metrics", {})

v_success = v_metrics.get("success_rate", 0)
nv_success = nv_metrics.get("success_rate", 0)

v_halluc = v_metrics.get("average_hallucinations_per_problem", 0)
nv_halluc = nv_metrics.get("average_hallucinations_per_problem", 0)

v_steps = v_metrics.get("average_success_steps", 0)
nv_steps = nv_metrics.get("average_success_steps", 0)

v_missing = v_metrics.get("total_missing_objects", 0)
nv_missing = nv_metrics.get("total_missing_objects", 0)

v_failed = v_metrics.get("total_failed_conditions", 0)
nv_failed = nv_metrics.get("total_failed_conditions", 0)

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│  Metric                    │  With Vision  │  No Vision  │ Diff   │
├────────────────────────────────────────────────────────────────────┤
│  Success Rate              │   {v_success:>7.1%}     │  {nv_success:>7.1%}    │ {(v_success-nv_success)*100:>+5.1f}% │
│  Avg Steps to Success      │   {v_steps:>7.2f}     │  {nv_steps:>7.2f}    │ {v_steps-nv_steps:>+5.2f}  │
│  Avg Hallucinations/Prob   │   {v_halluc:>7.2f}     │  {nv_halluc:>7.2f}    │ {v_halluc-nv_halluc:>+5.2f}  │
│  Total Missing Objects     │   {v_missing:>7d}     │  {nv_missing:>7d}    │ {v_missing-nv_missing:>+5d}  │
│  Total Failed Conditions   │   {v_failed:>7d}     │  {nv_failed:>7d}    │ {v_failed-nv_failed:>+5d}  │
└────────────────────────────────────────────────────────────────────┘
""")

# Conclusion
diff = v_success - nv_success
if diff > 0.05:
    print("📊 Conclusion: Vision HELPS - {:.1%} improvement in success rate".format(diff))
elif diff < -0.05:
    print("📊 Conclusion: Vision HURTS - {:.1%} decrease in success rate".format(abs(diff)))
else:
    print("📊 Conclusion: Vision has MINIMAL impact (within ±5%)")

EOF

else
    echo -e "${RED}Error: Missing result files for comparison${NC}"
fi

# Final summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Vision Comparison Complete${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Results saved to: $RUN_DIR"
echo ""

echo "Vision comparison completed at $(date)" >> "$LOG_FILE"


