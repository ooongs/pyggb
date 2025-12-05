# Problem Parser 업데이트 요약

`problem_parser.py` 파일에 요청하신 4가지 기능을 모두 추가했습니다.

## 추가된 기능

### 1. ✅ 모호한 참조 감지 및 스킵 (∠1, ∠2 등)

그림 없이는 정확히 어떤 것을 지칭하는지 알 수 없는 번호가 매겨진 각도(∠1, ∠2, ∠3 등)가 포함된 문제는 **자동으로 건너뜁니다**.

**구현 메서드:**

- `has_ambiguous_references(problem_text)` - 모호한 참조 감지
- `parse_problem(..., skip_ambiguous=True)` - 모호한 문제 자동 스킵

**예시:**

```python
# ❌ 이런 문제는 건너뜀
"如图所示,∠1=30°,∠2=45°,求∠3的大小"

# ✅ 이런 문제는 파싱됨
"如图,AB∥CD,∠ABC=50°,则∠BCD等于()"
```

### 2. ✅ 문제 텍스트 정리

기하학 객체를 그리는데 영향을 미치지 않는 구문들을 제거합니다:

**제거되는 구문:**

- `如图所示`, `如图`, `图中` 등 그림 참조 구문
- `则∠AOB的大小是()` - 값을 구하는 질문 부분
- `求...()`, `计算...()`, `证明...()` - 증명/계산 요구 부분

**구현 메서드:**

- `clean_problem_text(problem_text)` - 텍스트 정리

**예시:**

```python
원본: "如图,AB∥CD,∠EFG=50°,则∠EGF等于()"
정리: "AB∥CD,∠EFG=50°"
```

### 3. ✅ 기하학 문제 분류

LLM을 사용하여 문제를 10가지 카테고리로 자동 분류합니다:

**분류 카테고리:**

1. **Basic Constructions** (기본 구성)
2. **Circle Properties & Constructions** (원의 성질 및 구성)
3. **Geometric Transformations** (기하학적 변환)
4. **Triangle Properties & Constructions** (삼각형 성질 및 구성)
5. **Applications of Geometric Theorems** (기하학 정리의 응용)
6. **Polygon Properties & Constructions** (다각형 성질 및 구성)
7. **Measurement & Ratios** (측정 및 비율)
8. **Locus Constructions** (궤적 구성)
9. **Angle Relationships** (각도 관계)
10. **Similarity & Congruence** (닮음 및 합동)

**구현 메서드:**

- `classify_problem(problem_text)` - 문제 분류

**데이터에 추가되는 필드:**

```json
{
  "category": "Angle Relationships"
}
```

### 4. ✅ 기하학 그림 난이도 평가 (1-5)

LLM을 사용하여 기하학 그림의 구성 난이도를 1~5로 평가합니다:

**난이도 기준:**

- **1 = 매우 쉬움** - 기본 도형, 적은 객체, 단순한 관계
- **2 = 쉬움** - 단순한 구성, 명확한 관계
- **3 = 보통** - 중간 복잡도, 여러 객체
- **4 = 어려움** - 복잡한 구성, 많은 관계, 신중한 계획 필요
- **5 = 매우 어려움** - 매우 복잡, 많은 객체, 복잡한 관계

**평가 요소:**

- 기하학 객체의 수 (점, 선, 원, 다각형)
- 관계의 복잡도 (평행, 수직, 각도 등)
- 제약 조건의 수
- 특수 구성의 필요성 (이등분선, 수선 등)

**구현 메서드:**

- `rate_difficulty(problem_text)` - 난이도 평가

**데이터에 추가되는 필드:**

```json
{
  "difficulty": 3
}
```

## 사용 방법

### 개별 문제 파싱

```python
from problem_parser import ProblemParser, create_openai_api_function
import os

# OpenAI API 설정 (분류 및 난이도 평가에 필요)
api_key = os.getenv("OPENAI_API_KEY")
llm_func = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
parser = ProblemParser(llm_api_function=llm_func)

# 문제 파싱
result = parser.parse_problem(
    problem_text="如图,AB∥CD,∠ABC=50°,则∠BCD等于()",
    problem_id="1",
    skip_ambiguous=True,  # ∠1, ∠2 등이 있는 문제 건너뛰기
    clean_text=True       # "如图所示" 및 질문 부분 제거
)

if result:
    print(f"카테고리: {result['category']}")
    print(f"난이도: {result['difficulty']}/5")
    print(f"정리된 텍스트: {result['cleaned_text']}")
```

### 배치 처리 (데이터셋 생성)

```python
# 디렉토리의 모든 JSON 파일 처리
stats = parser.batch_parse_directory(
    input_dir="data-5/GeoQA3/json",
    output_file="ground_truth/geoqa3_dataset.json",
    skip_ambiguous=True,
    clean_text=True
)

print(f"파싱 성공: {stats['parsed']}")
print(f"건너뜀: {stats['skipped']}")
```

### 간단한 방법: create_dataset.py 스크립트 사용

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY='your-api-key-here'

# 데이터셋 생성 스크립트 실행
python create_dataset.py
```

## 출력 형식

파싱된 데이터는 다음과 같은 구조를 갖습니다:

```json
{
  "id": "1",
  "original_text": "如图,AB∥CD,...",
  "cleaned_text": "AB∥CD,...",
  "category": "Angle Relationships",
  "difficulty": 3,
  "required_objects": {
    "points": ["A", "B", "C", "D"],
    "segments": [["A", "B"], ["C", "D"]],
    ...
  },
  "verification_conditions": [...]
}
```

배치 처리 출력 (`batch_parse_directory`):

```json
{
  "metadata": {
    "total_files": 100,
    "parsed": 85,
    "skipped": 12,
    "errors": 3
  },
  "problems": [...]
}
```

## 제공된 파일

1. **problem_parser.py** (수정됨)

   - 모든 새 기능이 포함된 메인 파서

2. **create_dataset.py** (신규)

   - 데이터셋 생성을 위한 편리한 스크립트
   - 사용하기 전에 `input_dir`와 `output_file` 경로를 수정하세요

3. **test_parser_features.py** (신규)

   - 모든 새 기능을 테스트하는 데모 스크립트
   - 실행: `python test_parser_features.py`

4. **PROBLEM_PARSER_GUIDE.md** (신규)

   - 상세한 영문 사용 가이드

5. **PROBLEM_PARSER_UPDATE_KR.md** (이 파일)
   - 한국어 업데이트 요약

## 테스트 결과

```bash
# 기능 테스트 실행
python test_parser_features.py
```

**테스트 결과:**

- ✅ 모호한 참조 감지 (∠1, ∠2, ∠3) - 정상 작동
- ✅ 텍스트 정리 ("如图", "则...等于()") - 정상 작동
- ✅ 파싱 기능 - 정상 작동
- ✅ 분류 및 난이도 평가 - OpenAI API 키 필요

## 요구 사항

```bash
# 필요한 패키지 설치
pip install openai

# OpenAI API 키 설정
export OPENAI_API_KEY='your-api-key-here'
```

**참고:** OpenAI API 키 없이도 규칙 기반 파싱은 가능하지만, 분류와 난이도 평가는 사용할 수 없습니다.

## 예시

### 건너뛰어지는 문제들

```python
# ❌ 모호한 각도 참조
"如图所示,∠1=30°,∠2=45°,求∠3的大小"

# ❌ 이름 없는 번호 각도
"已知∠1∥∠2,∠3=60°"
```

### 성공적으로 파싱되는 문제들

```python
# ✅ 이름이 있는 각도와 점
"如图,AB∥CD,∠ABC=50°,则∠BCD等于()"
# 정리됨: "AB∥CD,∠ABC=50°"

# ✅ 삼각형 문제
"在三角形ABC中,AB=BC,∠ABC=90°"
# 정리됨: "在三角形ABC中,AB=BC,∠ABC=90°"
```

## 추가 정보

더 자세한 정보는 `PROBLEM_PARSER_GUIDE.md` (영문)를 참조하세요.

질문이나 문제가 있으면 알려주세요!


