#!/bin/bash
# Simple parallel processing example
# Run 0-500, 500-1000 ranges in parallel

# Configuration
INPUT_DIR="${INPUT_DIR:-data-5/GeoQA3/json}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
MODEL="${MODEL:-gpt-5-mini}"

echo "Parallel dataset creation started..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run multiple ranges in parallel in the background
echo "Range 0-500 started..."
python scripts/create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 1700 --end 1730 --model "$MODEL" > "$OUTPUT_DIR/log_0_500.log" 2>&1 &
PID1=$!

echo "Range 500-1000 started..."
python scripts/create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 1800 --end 1830 --model "$MODEL" > "$OUTPUT_DIR/log_500_1000.log" 2>&1 &
PID2=$!

# If more ranges are needed, uncomment the following lines
# echo "Range 1000-1500 started..."
# python scripts/create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 1000 --end 1500 --model "$MODEL" > "$OUTPUT_DIR/log_1000_1500.log" 2>&1 &
# PID3=$!

echo ""
echo "All processes are running in the background."
echo "Check progress:"
echo "  tail -f $OUTPUT_DIR/log_1500.log"
echo "  tail -f $OUTPUT_DIR/log_1550.log"
echo ""
echo "프로세스 ID: $PID1, $PID2"
echo ""

# Wait for all processes to complete
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?

echo ""
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "✓ All ranges processed successfully!"
    echo ""
    echo "Merge final dataset:"
    echo "  python scripts/create_dataset.py --merge"
else
    echo "✗ Some ranges failed"
    echo "  Range 0-500: $([ $EXIT1 -eq 0 ] && echo 'Success' || echo 'Failed')"
    echo "  Range 500-1000: $([ $EXIT2 -eq 0 ] && echo 'Success' || echo 'Failed')"
fi



