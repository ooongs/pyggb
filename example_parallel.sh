#!/bin/bash
# 병렬 처리 예제 - 사용자 정의 범위

# 이 스크립트를 복사해서 원하는 범위로 수정하세요

INPUT_DIR="data-5/GeoQA3/json"
OUTPUT_DIR="ground_truth"
MODEL="gpt-4.1"

# 범위 정의 (원하는 대로 수정)
RANGES=(
    "0:200"
    "200:300"
    "300:400"
    # "1500:2000"  # 필요하면 주석 해제
)

echo "병렬 처리 시작: ${#RANGES[@]}개 범위"
echo ""

# 각 범위를 백그라운드로 실행
pids=()
for range in "${RANGES[@]}"; do
    IFS=':' read -r start end <<< "$range"
    echo "범위 $start-$end 시작..."
    
    python create_dataset.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --start "$start" \
        --end "$end" \
        --model "$MODEL" \
        > "$OUTPUT_DIR/log_${start}_${end}.log" 2>&1 &
    
    pids+=($!)
done

echo ""
echo "모든 프로세스 실행 중..."
echo "프로세스 ID: ${pids[@]}"
echo ""

# 모든 프로세스 완료 대기
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "✓ 모든 범위 처리 완료!"
echo ""
echo "병합: python create_dataset.py --merge"



