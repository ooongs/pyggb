# 🔬 Geometry Agent Benchmark Guide

이 가이드는 여러 LLM 모델의 기하학 문제 해결 능력을 비교 평가하는 방법을 설명합니다.

## 📊 평가 지표

### 1. 성공률 (Success Rate)
- 전체 문제 중 성공적으로 해결된 문제의 비율
- 성공 기준: object_score ≥ 90% AND condition_score ≥ 90%

### 2. 스텝 분석 (Step Analysis)
- 성공까지 걸리는 평균 스텝 수
- 스텝별 성공 분포 (1스텝 성공, 2스텝 성공, ...)
- 효율성 지표로 사용

### 3. 할루시네이션 분석 (Hallucination Analysis) ⭐ 주요 지표
- **정의**: DSL 코드 실행 시 발생하는 에러 횟수
- **에러 유형**:
  - `undefined_reference`: 정의되지 않은 객체 참조
  - `syntax_error`: 문법 오류
  - `type_error`: 타입 불일치
  - `invalid_command`: 잘못된 명령어
  - `duplicate_error`: 중복 정의
  - `constraint_error`: 제약 조건 오류
- **측정값**:
  - 총 할루시네이션 횟수
  - 문제당 평균 할루시네이션 횟수
  - 할루시네이션 복구 스텝 수 (에러 후 정상 실행까지)

### 4. 객체 누락 분석 (Object Accuracy) ⭐ 주요 지표
- **정의**: 요구되는 기하학적 객체 중 누락된 객체 수
- **유형**:
  - `points`: 점
  - `segments`: 선분
  - `lines`: 직선
  - `circles`: 원
  - `polygons`: 다각형
- 낮을수록 좋음

### 5. 조건 실패 분석 (Condition Accuracy) ⭐ 주요 지표
- **정의**: 검증 조건 중 실패한 조건 수
- **조건 유형**:
  - `parallel`: 평행
  - `perpendicular`: 수직
  - `angle_value`: 각도 값
  - `angle_equality`: 각도 동치
  - `segment_equality`: 선분 동치
  - `collinear`: 공선점
  - `midpoint_of`: 중점
  - `angle_bisector`: 각의 이등분선
- 낮을수록 좋음

## 🚀 사용 방법

### 기본 사용법

```bash
# 단일 모델, 제한된 문제 수로 테스트
python run_agent_benchmark.py --batch --model gpt-4o --limit 10

# 전체 데이터셋 테스트
python run_agent_benchmark.py --batch --model gpt-4o --output results_gpt4o.json

# 특정 문제만 테스트
python run_agent_benchmark.py --problem-id 0 --model gpt-4o --verbose
```

### 여러 모델 비교 테스트

```bash
# 기본 모델들 테스트 (10개 문제)
./run_multi_model_benchmark.sh --limit 10

# 특정 모델들만 테스트
./run_multi_model_benchmark.sh --models "gpt-4o,gpt-4o-mini" --limit 20

# vLLM 모델 테스트
./run_multi_model_benchmark.sh \
    --api-base http://localhost:8000/v1 \
    --models "Qwen/Qwen2.5-VL-7B-Instruct" \
    --limit 10

# 전체 데이터셋 테스트
./run_multi_model_benchmark.sh --full --models "gpt-4o"
```

### 결과 분석

```bash
# 여러 결과 파일 비교
python analyze_benchmark_results.py results_gpt4o.json results_gpt4o_mini.json

# 디렉토리의 모든 결과 분석
python analyze_benchmark_results.py --dir benchmark_results/run_20241206_120000

# CSV 형식으로 출력 (스프레드시트용)
python analyze_benchmark_results.py results_*.json --format csv --output comparison.csv

# JSON 형식으로 출력 (프로그래밍용)
python analyze_benchmark_results.py results_*.json --format json --output comparison.json

# 문제별 상세 비교 포함
python analyze_benchmark_results.py results_*.json --per-problem
```

## 📁 출력 파일 구조

### 개별 결과 파일 (`results_*.json`)

```json
{
  "metadata": {
    "timestamp": "2024-12-06T12:00:00",
    "model": "gpt-4o",
    "dataset": "benchmark_geoqa3.json",
    "max_iterations": 5
  },
  "metrics": {
    "success_rate": 0.75,
    "average_success_steps": 2.5,
    "total_hallucinations": 15,
    "average_hallucinations_per_problem": 1.5,
    "missing_objects_by_type": {...},
    "failed_conditions_by_type": {...}
  },
  "detailed_analysis": {...},
  "results": [...],
  "problem_details": [...]
}
```

### 비교 보고서 출력 예시

```
════════════════════════════════════════════════════════════════════════════════
📊 BENCHMARK RESULTS COMPARISON REPORT
Generated: 2024-12-06 12:00:00
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
📈 1. SUCCESS RATE COMPARISON
────────────────────────────────────────────────────────────────────────────────
Model                                    Success Rate    Solved/Total   
----------------------------------------------------------------------
gpt-4o                                   80.0%           80/100         
gpt-4o-mini                              65.0%           65/100         

────────────────────────────────────────────────────────────────────────────────
🔥 3. HALLUCINATION ANALYSIS (DSL Execution Errors)
────────────────────────────────────────────────────────────────────────────────
Model                                    Total      Avg/Problem     Avg Recovery   
--------------------------------------------------------------------------------
gpt-4o                                   50         0.50            1.20           
gpt-4o-mini                              120        1.20            1.80           

────────────────────────────────────────────────────────────────────────────────
🏆 6. OVERALL RANKINGS
────────────────────────────────────────────────────────────────────────────────
  🥇 Best Success Rate:
    🥇 gpt-4o: 80.0%
    🥈 gpt-4o-mini: 65.0%
```

## ⚙️ vLLM 서버 설정

### vLLM 서버 시작

```bash
# 기본 Vision-Language 모델
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 멀티 GPU
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4
```

### 환경 변수 설정

```bash
# .env 파일 또는 export
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy-key"  # vLLM은 API 키 검증 안함
```

## 📈 결과 해석 가이드

### 좋은 모델의 특성

| 지표 | 좋음 | 보통 | 나쁨 |
|------|------|------|------|
| 성공률 | > 80% | 50-80% | < 50% |
| 평균 스텝 | < 2 | 2-4 | > 4 |
| 할루시네이션/문제 | < 0.5 | 0.5-2.0 | > 2.0 |
| 복구 스텝 | < 1.5 | 1.5-3.0 | > 3.0 |
| 누락 객체 비율 | < 5% | 5-15% | > 15% |
| 실패 조건 비율 | < 10% | 10-30% | > 30% |

### 주요 분석 포인트

1. **성공률 vs 효율성**: 높은 성공률이지만 많은 스텝이 필요한 모델 vs 낮은 성공률이지만 효율적인 모델

2. **할루시네이션 패턴**: 특정 에러 유형이 많으면 해당 영역의 학습이 부족함을 의미
   - `undefined_reference`가 많음 → 객체 정의 순서 이해 부족
   - `syntax_error`가 많음 → DSL 문법 이해 부족

3. **객체 누락 패턴**: 특정 객체 유형이 많이 누락되면 해당 개념 이해 부족
   - `polygons` 누락 많음 → 다각형 구성 방법 이해 부족

4. **조건 실패 패턴**: 특정 조건이 자주 실패하면 해당 기하학적 관계 이해 부족
   - `parallel` 실패 많음 → 평행 관계 구현 어려움

## 🔧 고급 설정

### benchmark_config.yaml

```yaml
defaults:
  dataset: "benchmark_geoqa3.json"
  max_iterations: 5
  save_images: true

models:
  - name: "gpt-4o"
    provider: "openai"
    enabled: true
    
  - name: "Qwen/Qwen2.5-VL-7B-Instruct"
    provider: "vllm"
    api_base: "http://localhost:8000/v1"
    enabled: true
```

### 커스텀 분석 스크립트

```python
from analyze_benchmark_results import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer()
analyzer.load_result("results_gpt4o.json")
analyzer.load_result("results_qwen.json")

# 커스텀 비교
comparison = analyzer.compare_models()

# 특정 지표 추출
for model in comparison["models"]:
    print(f"{model['model']}: {model['success_rate']:.1%}")
```

## 📋 체크리스트

벤치마크 실행 전 확인사항:

- [ ] 필요한 API 키 설정 (.env 파일)
- [ ] 데이터셋 파일 존재 확인
- [ ] vLLM 서버 실행 중 (vLLM 모델 사용 시)
- [ ] 충분한 GPU 메모리 (Vision 모델은 많은 메모리 필요)
- [ ] 출력 디렉토리 쓰기 권한

## 🐛 문제 해결

### 일반적인 오류

1. **API 키 오류**
   ```
   ERROR: OPENAI_API_KEY not found
   ```
   → `.env` 파일에 API 키 설정

2. **vLLM 연결 오류**
   ```
   Connection refused
   ```
   → vLLM 서버가 실행 중인지 확인

3. **메모리 부족**
   ```
   CUDA out of memory
   ```
   → 더 작은 모델 사용 또는 GPU 메모리 정리

4. **모델 미발견**
   ```
   Model not found
   ```
   → vLLM 서버에 올바른 모델이 로드되었는지 확인



