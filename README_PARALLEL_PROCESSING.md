# Parallel Dataset Creation Guide

병렬 처리를 통해 대량의 기하학 문제를 빠르게 처리하는 방법을 설명합니다.

## 🚀 빠른 시작

### 방법 1: Bash 스크립트 사용 (가장 간단)

```bash
# 기본 설정으로 실행 (4 workers, 5000 files)
./run_parallel_dataset.sh

# 커스텀 설정으로 실행
./run_parallel_dataset.sh 8 5000 625  # 8 workers, 5000 files, 625 per batch
```

### 방법 2: 수동으로 여러 터미널에서 실행

```bash
# 터미널 1
python create_dataset.py --start 0 --end 1000

# 터미널 2
python create_dataset.py --start 1000 --end 2000

# 터미널 3
python create_dataset.py --start 2000 --end 3000

# 터미널 4
python create_dataset.py --start 3000 --end 4000

# 터미널 5
python create_dataset.py --start 4000 --end 5007

# 모든 터미널 완료 후 merge
python create_dataset.py --merge
```

---

## 📖 상세 사용법

### create_dataset.py 옵션

```bash
python create_dataset.py [OPTIONS]

Options:
  --input-dir DIR      입력 디렉토리 (default: data-5/GeoQA3/json)
  --output-dir DIR     출력 디렉토리 (default: ground_truth)
  --start N            시작 인덱스 (inclusive)
  --end N              끝 인덱스 (exclusive)
  --merge              모든 JSONL 파일을 하나의 JSON으로 병합
  --no-resume          이미 처리된 ID를 다시 처리
  --model MODEL        사용할 OpenAI 모델 (default: gpt-4.1-mini)
```

### 예시

```bash
# 전체 파일 처리 (순차적)
python create_dataset.py

# 범위 지정 처리 (0~500)
python create_dataset.py --start 0 --end 500

# 범위 지정 처리 (500~1000)
python create_dataset.py --start 500 --end 1000

# 커스텀 출력 디렉토리
python create_dataset.py --start 0 --end 100 --output-dir my_output

# resume 비활성화 (처음부터 다시 처리)
python create_dataset.py --start 0 --end 100 --no-resume

# 모든 JSONL 파일 병합
python create_dataset.py --merge --output-dir ground_truth
```

---

## 🔄 동작 방식

### 1. 증분 저장 (Incremental Saving)

각 문제를 파싱할 때마다 즉시 JSONL 파일에 저장합니다:

```
ground_truth/
├── dataset_0_1000.jsonl      # Worker 1 결과
├── dataset_1000_2000.jsonl   # Worker 2 결과
├── dataset_2000_3000.jsonl   # Worker 3 결과
└── ...
```

**장점:**

- 중간에 프로세스가 중단되어도 데이터 손실 없음
- Resume 모드로 중단된 지점부터 재개 가능
- 병렬 처리 시 각 worker가 독립적으로 파일에 기록

### 2. Resume 모드

출력 디렉토리의 **모든** JSONL 파일을 검사하여 이미 처리된 ID를 스킵합니다:

```bash
# 첫 번째 실행 (0~500 처리)
python create_dataset.py --start 0 --end 500
# → 500개 처리됨

# 두 번째 실행 (0~1000으로 확장)
python create_dataset.py --start 0 --end 1000
# → 기존 500개는 스킵, 새로운 500개만 처리
```

### 3. 병합 (Merge)

모든 JSONL 파일을 하나의 JSON 데이터셋으로 병합합니다:

```bash
python create_dataset.py --merge
```

**병합 결과:**

- 중복 ID 자동 제거
- 통계 자동 생성 (카테고리/난이도 분포)
- 최종 JSON 파일 생성

---

## 📁 출력 파일 형식

### JSONL 파일 (중간 결과)

각 줄이 하나의 JSON 객체:

```jsonl
{"id": "0", "status": "parsed", "category": "Triangle", ...}
{"id": "1", "status": "parsed", "category": "Circle", ...}
{"id": "22", "status": "skipped", "reason": "ambiguous_reference"}
{"id": "33", "status": "error", "error": "Connection error"}
```

### JSON 파일 (최종 결과)

```json
{
  "metadata": {
    "created_at": "2024-12-05T22:08:00",
    "total_problems": 4000,
    "skipped": 800,
    "errors": 7,
    "skipped_ids": ["22", "45", ...],
    "error_ids": ["33", ...]
  },
  "problems": [
    {
      "id": "0",
      "original_text": "...",
      "cleaned_text": "...",
      "category": "Triangle Properties & Constructions",
      "difficulty": 3,
      "required_objects": {...},
      "verification_conditions": [...]
    },
    ...
  ]
}
```

---

## 🛠️ 병렬 처리 전략

### 권장 설정

| 총 파일 수 | Workers | Batch Size | 예상 시간 (API) |
| ---------- | ------- | ---------- | --------------- |
| 1,000      | 2       | 500        | ~30분           |
| 5,000      | 4       | 1,250      | ~2시간          |
| 5,000      | 8       | 625        | ~1시간          |
| 10,000     | 8       | 1,250      | ~2시간          |

**참고:** 시간은 네트워크 속도와 API 응답 시간에 따라 달라집니다.

### Bash 스크립트 사용

```bash
# 기본 실행
./run_parallel_dataset.sh

# 8 workers로 5007개 파일 처리
./run_parallel_dataset.sh 8 5007

# 커스텀 batch size
./run_parallel_dataset.sh 4 5000 500
```

### 수동 실행 (더 많은 제어)

여러 터미널에서 동시 실행:

```bash
# Terminal 1
python create_dataset.py --start 0 --end 1250

# Terminal 2
python create_dataset.py --start 1250 --end 2500

# Terminal 3
python create_dataset.py --start 2500 --end 3750

# Terminal 4
python create_dataset.py --start 3750 --end 5007

# 완료 후 병합
python create_dataset.py --merge
```

---

## ⚠️ 주의사항

### 1. API Rate Limits

OpenAI API는 rate limit이 있습니다. 너무 많은 workers를 사용하면 rate limit에 걸릴 수 있습니다.

**해결 방법:**

- Workers 수를 줄이거나
- 각 worker 사이에 딜레이 추가

### 2. 파일 잠금 (File Locking)

JSONL 파일에 쓸 때 `fcntl.flock`을 사용하여 동시 쓰기를 방지합니다.
같은 범위를 여러 worker가 처리하면 중복이 발생할 수 있으므로, 범위가 겹치지 않도록 주의하세요.

### 3. 메모리 사용

병합 시 모든 JSONL 파일을 메모리에 로드합니다. 매우 큰 데이터셋의 경우 메모리 부족이 발생할 수 있습니다.

---

## 🔧 문제 해결

### 중단 후 재개

```bash
# Resume 모드는 기본적으로 활성화되어 있음
python create_dataset.py --start 0 --end 5000
# 중단됨...

# 다시 실행하면 이미 처리된 것은 자동으로 스킵
python create_dataset.py --start 0 --end 5000
```

### 특정 범위만 재처리

```bash
# --no-resume 옵션 사용
python create_dataset.py --start 1000 --end 2000 --no-resume
```

### 로그 확인

Bash 스크립트 사용 시 로그는 `logs/` 디렉토리에 저장됩니다:

```bash
# 로그 확인
tail -f logs/worker_0_0_1250.log
tail -f logs/worker_1_1250_2500.log
```

### API 오류

API 오류 발생 시 자동으로 rule-based 파싱으로 fallback됩니다.
오류가 기록된 ID는 병합 시 `error_ids`에 포함됩니다.

---

## 📊 결과 확인

### 병합 후 통계 확인

```bash
python create_dataset.py --merge
```

출력 예시:

```
============================================================
Merge Complete!
============================================================
  Total parsed: 4000
  Total skipped: 800
  Total errors: 7
  Output: ground_truth/geoqa3_dataset.json

Category Distribution:
  Triangle Properties & Constructions     : 1200 (30.0%)
  Angle Relationships                     :  800 (20.0%)
  Circle Properties & Constructions       :  600 (15.0%)
  ...

Difficulty Distribution:
  Level 1: ██████ (600)
  Level 2: ██████████████ (1400)
  Level 3: ████████████ (1200)
  Level 4: ████████ (600)
  Level 5: ████ (200)
```

### JSON 파일 분석

```python
import json

with open('ground_truth/geoqa3_dataset.json', 'r') as f:
    data = json.load(f)

print(f"Total problems: {len(data['problems'])}")
print(f"Skipped: {data['metadata']['skipped']}")
print(f"Errors: {data['metadata']['errors']}")
```

---

## 🎯 최적의 워크플로우

1. **준비:**

   ```bash
   export OPENAI_API_KEY='your-key'
   ```

2. **병렬 처리 실행:**

   ```bash
   ./run_parallel_dataset.sh 4 5007
   ```

3. **진행 상황 확인:**

   ```bash
   # 로그 확인
   tail -f logs/worker_*.log

   # 처리된 파일 수 확인
   wc -l ground_truth/dataset_*.jsonl
   ```

4. **병합:**

   ```bash
   python create_dataset.py --merge
   ```

5. **결과 확인:**
   ```bash
   cat ground_truth/geoqa3_dataset.json | python -m json.tool | head -50
   ```

---

## 💡 팁

1. **작은 테스트 먼저:** 전체 처리 전에 작은 범위로 테스트하세요.

   ```bash
   python create_dataset.py --start 0 --end 100
   ```

2. **로그 모니터링:** 병렬 실행 시 로그를 주시하세요.

   ```bash
   watch -n 5 'wc -l ground_truth/dataset_*.jsonl'
   ```

3. **야간 실행:** 대량 처리는 야간에 실행하는 것이 좋습니다.

4. **백업:** 중요한 결과는 병합 후 백업하세요.
   ```bash
   cp ground_truth/geoqa3_dataset.json ground_truth/geoqa3_dataset_backup.json
   ```



