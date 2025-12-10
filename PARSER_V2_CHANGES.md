# Problem Parser V2 - 주요 변경사항

## 🎯 변경 요약

### 1. 두 단계 API 호출 분리

이제 LLM API를 **두 번** 호출합니다:

| 단계        | 목적                           | 내용                               |
| ----------- | ------------------------------ | ---------------------------------- |
| **Stage 1** | 문제 적합도 판별 + 텍스트 정리 | 정의되지 않은 점, 모호한 각도 감지 |
| **Stage 2** | 객체/조건 추출 + 난이도 평가   | **작도 관점** 난이도 판단          |

### 2. 강화된 문제 필터링

**기존**: ∠1, ∠2 같은 번호 각도만 필터링

**변경 후**: 다음 모든 경우 필터링

| 유형             | 예시       | 이유                        |
| ---------------- | ---------- | --------------------------- |
| 정의되지 않은 점 | `∠E=40°`   | E가 어느 선 위인지 모름     |
| 위치 미정 점     | `∠BDC=30°` | D가 어디에 있는지 정의 안됨 |
| 단일 문자 각도   | `∠D=26°`   | ∠ADB? ∠BDC? 알 수 없음      |
| 번호 각도        | `∠1=30°`   | 그림 없이 알 수 없음        |

### 3. 텍스트 정리 강화

**제거되는 항목:**

```
(3分)                    → 제거 (점수)
如图所示                  → 제거 (그림 참조)
那么∠BOD为()            → 제거 (질문)
则∠ABC的大小是()        → 제거 (질문)
```

### 4. 난이도 평가 기준 변경

**기존**: 문제 풀이 관점 (어려운 계산 = 높은 난이도)

**변경**: **작도 관점** (그림 그리기 어려움 = 높은 난이도)

| 난이도 | 설명        | 예시                     |
| ------ | ----------- | ------------------------ |
| 1      | 매우 쉬움   | 단일 삼각형, 직사각형    |
| 2      | 쉬움        | 기본 구성, 적은 조건     |
| 3      | 보통        | 여러 객체, 몇 가지 관계  |
| 4      | 어려움      | 복잡한 구성, 많은 객체   |
| 5      | 매우 어려움 | 여러 원, 접선, 동시점 등 |

**작도 난이도 평가 요소:**

- 점, 선, 원의 개수
- 의존적 구성 (교점에 의존하는 점 등)
- 정밀도 요구 (특정 각도, 길이)
- 보조 구성 필요성

---

## 📋 API 호출 구조

### Stage 1: 검증 및 정리

```python
validate_and_clean_problem(problem_text, problem_id)
# Returns: (is_valid, cleaned_text, rejection_reason)
```

**검증 항목:**

- 모든 점이 정의되어 있는가?
- 모든 각도가 명확한가?
- 구성 가능한 조건이 충분한가?

**정리 항목:**

- 점수 제거: `(3分)`
- 그림 참조 제거: `如图所示`
- 질문 제거: `则∠AOB的大小是()`

### Stage 2: 파싱 및 평가

```python
_parse_and_rate_with_llm(cleaned_text, problem_id)
# Returns: parsed_data with category and difficulty
```

**추출 항목:**

- `required_objects`: 점, 선분, 원, 다각형
- `verification_conditions`: 조건들
- `category`: 분류
- `difficulty`: 작도 난이도 (1-5)

---

## 🧪 테스트 예시

### 통과해야 하는 문제

```python
# ✓ 모든 점이 정의됨
"如图,AB∥CD,直线EF交AB于点E,交CD于点F,∠EFG=50°"

# ✓ 삼각형 컨텍스트에서 ∠A는 명확
"在△ABC中,∠A=60°,AB=BC"

# ✓ D의 위치가 정의됨
"在△ABC中,D在AB上,∠ACD=∠B"
```

### 거부되어야 하는 문제

```python
# ❌ E의 위치 미정
"AB∥CD,∠E=40°,∠A=110°"

# ❌ D의 위치 미정 (어느 선 위?)
"在△ABC中,∠C=90°,∠BDC=30°,AD=2BC"

# ❌ 단일 문자 각도
"AB∥CD,∠D=26°,∠E=35°"

# ❌ 번호 각도
"∠1=30°,∠2=45°"
```

---

## 🚀 사용 방법

### 단일 문제 파싱

```python
from problem_parser import ProblemParser, create_openai_api_function

llm_func = create_openai_api_function(model="gpt-4o-mini")
parser = ProblemParser(llm_api_function=llm_func)

# 두 단계 API 호출이 자동으로 수행됨
result = parser.parse_problem(
    problem_text="在△ABC中,D在AB上,∠ACD=∠B,CD=4",
    problem_id="1"
)

if result:
    print(f"Category: {result['category']}")
    print(f"Construction Difficulty: {result['difficulty']}/5")
else:
    print("Problem rejected (validation failed)")
```

### 배치 처리

```bash
# 범위 지정 처리
python create_dataset.py --start 0 --end 500

# 병렬 처리
./run_parallel_simple.sh

# 병합
python create_dataset.py --merge
```

---

## 📊 출력 형식

```json
{
  "id": "1",
  "original_text": "如图,AB∥CD,...",
  "cleaned_text": "AB∥CD,...",
  "category": "Angle Relationships",
  "difficulty": 3,
  "required_objects": {
    "points": ["A", "B", "C", "D"],
    "segments": [
      ["A", "B"],
      ["C", "D"]
    ],
    "lines": [],
    "circles": [],
    "polygons": []
  },
  "verification_conditions": [
    {
      "type": "parallel",
      "objects": [
        ["A", "B"],
        ["C", "D"]
      ]
    }
  ]
}
```

---

## ⚠️ 중요 변경사항

1. **API 호출 2배**: 각 문제당 2번의 API 호출 (검증 + 파싱)
2. **거부율 증가**: 더 엄격한 검증으로 거부되는 문제 증가 예상
3. **난이도 의미 변경**: 문제 풀이 → 작도 관점
4. **clean_text 파라미터 무시**: Stage 1에서 자동 처리

---

## 📁 변경된 파일

- `problem_parser.py` - 두 단계 처리, 새 검증 로직
- `create_dataset.py` - 진행 상황 출력 개선
- `run_parallel_simple.sh` - 모델명 수정

---

## 🔍 디버깅

### 거부 이유 확인

```python
is_valid, cleaned, reason = parser.validate_and_clean_problem(text, "test")
if not is_valid:
    print(f"Rejected: {reason}")
```

### API 응답 확인

Stage 1 API는 다음 JSON을 반환:

```json
{
  "is_valid": false,
  "cleaned_text": "",
  "rejection_reason": "Point E is referenced in ∠E but its position is not defined"
}
```



