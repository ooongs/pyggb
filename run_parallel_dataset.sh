#!/bin/bash
# 병렬 데이터셋 생성 스크립트
# 여러 범위를 동시에 처리하여 속도 향상

set -e  # 에러 발생 시 중단

# 설정
INPUT_DIR="${INPUT_DIR:-data-5/GeoQA3/json}"
OUTPUT_DIR="${OUTPUT_DIR:-ground_truth}"
BATCH_SIZE="${BATCH_SIZE:-100}"  # 각 범위의 크기
MAX_PARALLEL="${MAX_PARALLEL:-4}"  # 동시 실행할 최대 프로세스 수
MODEL="${MODEL:-gpt-4.1}"

# 색상 출력
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}병렬 데이터셋 생성 스크립트${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 입력 디렉토리 확인
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ 오류: 입력 디렉토리를 찾을 수 없습니다: $INPUT_DIR${NC}"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# JSON 파일 개수 확인
TOTAL_FILES=$(find "$INPUT_DIR" -name "*.json" | wc -l | tr -d ' ')
echo "총 파일 수: $TOTAL_FILES"
echo "배치 크기: $BATCH_SIZE"
echo "최대 병렬 프로세스: $MAX_PARALLEL"
echo ""

# OpenAI API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  경고: OPENAI_API_KEY가 설정되지 않았습니다.${NC}"
    echo "   분류 및 난이도 평가는 사용할 수 없습니다."
    echo ""
    read -p "계속하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 범위 계산
NUM_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "총 배치 수: $NUM_BATCHES"
echo ""

# 로그 디렉토리 생성
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# PID 파일 디렉토리
PID_DIR="$OUTPUT_DIR/pids"
mkdir -p "$PID_DIR"

# 각 배치를 처리하는 함수
process_batch() {
    local start=$1
    local end=$2
    local batch_num=$3
    
    local log_file="$LOG_DIR/batch_${start}_${end}.log"
    local pid_file="$PID_DIR/batch_${start}_${end}.pid"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 배치 $batch_num 시작: $start ~ $end" | tee -a "$log_file"
    
    python create_dataset.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --start "$start" \
        --end "$end" \
        --model "$MODEL" \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 배치 $batch_num 완료: $start ~ $end" | tee -a "$log_file"
        rm -f "$pid_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 배치 $batch_num 실패: $start ~ $end (exit code: $exit_code)" | tee -a "$log_file"
        rm -f "$pid_file"
        return $exit_code
    fi
}

# 병렬 실행 관리
declare -a pids=()
declare -a batch_starts=()
declare -a batch_ends=()
declare -a batch_nums=()

batch_num=0
for ((start=0; start<TOTAL_FILES; start+=BATCH_SIZE)); do
    end=$((start + BATCH_SIZE))
    if [ $end -gt $TOTAL_FILES ]; then
        end=$TOTAL_FILES
    fi
    
    batch_num=$((batch_num + 1))
    
    # 최대 병렬 프로세스 수까지 대기
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        # 완료된 프로세스 확인
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # 프로세스 완료
                wait "${pids[$i]}"
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo -e "${RED}✗ 배치 ${batch_nums[$i]} 실패 (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
                else
                    echo -e "${GREEN}✓ 배치 ${batch_nums[$i]} 완료 (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
                fi
                unset pids[$i]
                unset batch_starts[$i]
                unset batch_ends[$i]
                unset batch_nums[$i]
            fi
        done
        
        # 배열 재인덱싱
        pids=("${pids[@]}")
        batch_starts=("${batch_starts[@]}")
        batch_ends=("${batch_ends[@]}")
        batch_nums=("${batch_nums[@]}")
        
        sleep 1
    done
    
    # 새 배치 시작
    echo -e "${YELLOW}→ 배치 $batch_num 시작: $start ~ $end${NC}"
    process_batch "$start" "$end" "$batch_num" &
    pid=$!
    
    pids+=($pid)
    batch_starts+=($start)
    batch_ends+=($end)
    batch_nums+=($batch_num)
    
    # PID 저장
    echo $pid > "$PID_DIR/batch_${start}_${end}.pid"
done

# 모든 프로세스 완료 대기
echo ""
echo "모든 배치가 시작되었습니다. 완료를 기다리는 중..."
echo ""

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}✗ 배치 ${batch_nums[$i]} 실패 (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
    else
        echo -e "${GREEN}✓ 배치 ${batch_nums[$i]} 완료 (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}모든 배치 처리 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "최종 데이터셋을 병합하려면:"
echo "  python create_dataset.py --merge"
echo ""
echo "로그 파일 위치: $LOG_DIR"
echo ""



