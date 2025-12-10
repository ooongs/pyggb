#!/bin/bash
# 간단한 병렬 처리 예제
# 0-500, 500-1000 범위를 동시에 실행

# 설정
INPUT_DIR="${INPUT_DIR:-data-5/GeoQA3/json}"
OUTPUT_DIR="${OUTPUT_DIR:-ground_truth}"
MODEL="${MODEL:-gpt-4.1-mini}"

echo "병렬 데이터셋 생성 시작..."
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 백그라운드로 여러 범위를 동시에 실행
echo "범위 0-500 시작..."
python create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 0 --end 300 --model "$MODEL" > "$OUTPUT_DIR/log_0_500.log" 2>&1 &
PID1=$!

echo "범위 500-1000 시작..."
python create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 300 --end 600 --model "$MODEL" > "$OUTPUT_DIR/log_500_1000.log" 2>&1 &
PID2=$!

# 더 많은 범위가 필요하면 아래 주석을 해제하세요
# echo "범위 1000-1500 시작..."
# python create_dataset.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --start 1000 --end 1500 --model "$MODEL" > "$OUTPUT_DIR/log_1000_1500.log" 2>&1 &
# PID3=$!

echo ""
echo "모든 프로세스가 백그라운드에서 실행 중입니다."
echo "진행 상황 확인:"
echo "  tail -f $OUTPUT_DIR/log_0_500.log"
echo "  tail -f $OUTPUT_DIR/log_500_1000.log"
echo ""
echo "프로세스 ID: $PID1, $PID2"
echo ""

# 모든 프로세스 완료 대기
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?

echo ""
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "✓ 모든 범위 처리 완료!"
    echo ""
    echo "최종 데이터셋 병합:"
    echo "  python create_dataset.py --merge"
else
    echo "✗ 일부 범위에서 오류 발생"
    echo "  범위 0-500: $([ $EXIT1 -eq 0 ] && echo '성공' || echo '실패')"
    echo "  범위 500-1000: $([ $EXIT2 -eq 0 ] && echo '성공' || echo '실패')"
fi



