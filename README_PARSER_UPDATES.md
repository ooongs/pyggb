# Problem Parser Updates 📚

## 🎯 요약 (Summary)

`problem_parser.py`에 요청하신 4가지 핵심 기능을 모두 구현했습니다:

1. ✅ **∠1, ∠2 등 모호한 참조 자동 스킵**
2. ✅ **"如图所示", 질문 부분 등 텍스트 자동 정리**
3. ✅ **기하학 문제 자동 분류 (10가지 카테고리)**
4. ✅ **기하학 그림 난이도 평가 (1-5 단계)**

---

## 🚀 빠른 시작 (Quick Start)

### 방법 1: 데모 실행 (권장)

```bash
# 샘플 20개 파일로 모든 기능 테스트
python example_batch_process.py
```

### 방법 2: 기능 테스트

```bash
# 모든 새 기능 개별 테스트
python test_parser_features.py
```

### 방법 3: 전체 데이터셋 생성

```bash
# OpenAI API 키 설정 (분류 및 난이도 평가용)
export OPENAI_API_KEY='your-api-key-here'

# create_dataset.py에서 경로 수정 후 실행
python create_dataset.py
```

---

## 📁 파일 구조

### 수정된 파일

- **`problem_parser.py`** - 모든 새 기능이 추가된 메인 파서

### 새로 생성된 파일

1. **`create_dataset.py`** - 데이터셋 생성 스크립트
2. **`example_batch_process.py`** - 배치 처리 데모 (샘플 20개)
3. **`test_parser_features.py`** - 기능 테스트 스크립트

### 문서 파일

1. **`PROBLEM_PARSER_GUIDE.md`** - 상세 영문 가이드
2. **`PROBLEM_PARSER_UPDATE_KR.md`** - 한국어 업데이트 설명
3. **`QUICK_START_PARSER.md`** - 빠른 참조 가이드
4. **`CHANGES_SUMMARY.md`** - 변경 사항 요약
5. **`README_PARSER_UPDATES.md`** - 이 파일

---

## 🎨 주요 기능 상세

### 1️⃣ 모호한 참조 감지 및 스킵

**문제점:** 그림 없이는 이해할 수 없는 번호 각도 (∠1, ∠2, ∠3)

**해결책:** 자동으로 감지하고 데이터셋에서 제외

**예시:**

```python
# ❌ 건너뛰어짐
"如图所示,∠1=30°,∠2=45°,求∠3的大小"

# ✅ 파싱됨
"如图,AB∥CD,∠ABC=50°,则∠BCD等于()"
```

**테스트 결과:**

- 첫 50개 파일 중 9개(18%) 스킵
- 전체 데이터셋에서 예상 스킵률: 10-20%

### 2️⃣ 텍스트 정리

**제거되는 내용:**

- 그림 참조: `如图所示`, `如图`, `图中`
- 질문 부분: `则∠AOB的大小是()`, `求...()`, `计算...()`, `证明...()`

**예시:**

```
입력: "如图,AB∥CD,直线EF交AB于点E,交CD于点F,∠EFG=50°,则∠EGF等于()"
출력: "AB∥CD,直线EF交AB于点E,交CD于点F,∠EFG=50°"
```

### 3️⃣ 문제 분류

**10가지 카테고리:**

1. Basic Constructions (기본 구성)
2. Circle Properties & Constructions (원)
3. Geometric Transformations (변환)
4. Triangle Properties & Constructions (삼각형)
5. Applications of Geometric Theorems (정리 응용)
6. Polygon Properties & Constructions (다각형)
7. Measurement & Ratios (측정/비율)
8. Locus Constructions (궤적)
9. Angle Relationships (각도 관계)
10. Similarity & Congruence (닮음/합동)

**요구사항:** OpenAI API 키 필요

### 4️⃣ 난이도 평가

**난이도 레벨:**

- **1** = 매우 쉬움 (기본 도형, 적은 객체)
- **2** = 쉬움 (단순 구성)
- **3** = 보통 (중간 복잡도)
- **4** = 어려움 (복잡한 구성)
- **5** = 매우 어려움 (매우 복잡)

**평가 기준:**

- 객체 수 (점, 선, 원, 다각형)
- 관계 복잡도 (평행, 수직, 각도)
- 제약 조건 수
- 특수 구성 필요성

**요구사항:** OpenAI API 키 필요

---

## 💻 코드 예시

### 기본 사용

```python
from problem_parser import ProblemParser, create_openai_api_function
import os

# API 키 설정 (선택사항, 분류/난이도 평가용)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    llm_func = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
    parser = ProblemParser(llm_api_function=llm_func)
else:
    parser = ProblemParser()  # 규칙 기반 파싱만

# 단일 문제 파싱
result = parser.parse_problem(
    problem_text="如图,AB∥CD,∠ABC=50°",
    problem_id="1",
    skip_ambiguous=True,  # ∠1, ∠2 등 스킵
    clean_text=True       # 텍스트 정리
)

if result:
    print(f"분류: {result.get('category', 'N/A')}")
    print(f"난이도: {result.get('difficulty', 'N/A')}/5")
    print(f"정리된 텍스트: {result['cleaned_text']}")
```

### 배치 처리

```python
# 전체 디렉토리 처리
stats = parser.batch_parse_directory(
    input_dir="data-5/GeoQA3/json",
    output_file="dataset.json",
    skip_ambiguous=True,
    clean_text=True
)

print(f"성공: {stats['parsed']}, 스킵: {stats['skipped']}")
```

---

## 📊 출력 형식

```json
{
  "metadata": {
    "total_files": 100,
    "parsed": 85,
    "skipped": 12,
    "errors": 3
  },
  "problems": [
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
        },
        { "type": "angle_value", "points": [["A", "B", "C"]], "value": 50 }
      ]
    }
  ]
}
```

---

## 🧪 테스트 결과

### 샘플 20개 파일 테스트

```
Total files: 20
Successfully parsed: 16 (80%)
Skipped: 4 (20%)
Errors: 0 (0%)
```

**건너뛴 파일들:**

- 1000.json: `∠1=35°,则∠2的度数为()`
- 1004.json: `∠2+∠3+∠4=320°,则∠1=()`
- 1007.json: `∠1=25°,则∠2的度数是()`
- 1017.json: `若∠1=30°,则∠2的度数是()`

모두 ∠1, ∠2 등 번호 각도 포함 → 정상적으로 스킵됨 ✓

---

## 📦 설치 및 설정

### 필수 요구사항

```bash
# OpenAI 패키지 설치
pip install openai
```

### OpenAI API 키 설정 (선택사항)

```bash
# 환경 변수로 설정
export OPENAI_API_KEY='your-api-key-here'

# 또는 ~/.zshrc에 추가
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.zshrc
source ~/.zshrc
```

**참고:** API 키 없이도 사용 가능하지만, 분류와 난이도 평가는 사용할 수 없습니다.

---

## 🎯 사용 시나리오

### 시나리오 1: 깨끗한 데이터셋 생성

```bash
# 모호한 문제 제거, 텍스트 정리
python create_dataset.py
```

### 시나리오 2: 난이도별 필터링

```python
# 쉬운 문제만 선택 (난이도 1-3)
result = parser.parse_problem(text)
if result and result.get('difficulty', 0) <= 3:
    use_problem(result)
```

### 시나리오 3: 카테고리별 그룹화

```python
# 카테고리별로 문제 분류
categories = {}
for problem in dataset['problems']:
    cat = problem['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(problem)
```

---

## 💡 유용한 팁

1. **API 키는 꼭 설정하세요** - 분류와 난이도 평가에 필수
2. **먼저 샘플로 테스트** - `example_batch_process.py` 실행
3. **스킵 비율 확인** - 너무 많이 스킵되면 `skip_ambiguous=False` 고려
4. **카테고리 커스터마이징** - `ProblemParser.PROBLEM_CATEGORIES` 수정
5. **배치 처리 권장** - 큰 데이터셋은 `batch_parse_directory()` 사용

---

## 📚 문서 가이드

- **빠르게 시작**: `QUICK_START_PARSER.md`
- **상세 가이드**: `PROBLEM_PARSER_GUIDE.md` (영문)
- **업데이트 요약**: `PROBLEM_PARSER_UPDATE_KR.md` (한글)
- **변경 사항**: `CHANGES_SUMMARY.md`

---

## ✅ 체크리스트

데이터셋 생성 전 확인사항:

- [ ] OpenAI API 키 설정 완료
- [ ] `create_dataset.py`에서 경로 확인/수정
- [ ] 샘플 테스트 실행 (`example_batch_process.py`)
- [ ] 모든 기능 테스트 (`test_parser_features.py`)
- [ ] 출력 디렉토리 존재 확인

---

## 🎉 완료된 작업

✅ 모든 요청 기능 구현 완료
✅ 테스트 스크립트 작성 완료
✅ 문서화 완료 (한글/영문)
✅ 배치 처리 기능 추가
✅ 샘플 데이터로 검증 완료

---

## 📞 문제 해결

### API 키 오류

```
Error: OpenAI API key not found
→ export OPENAI_API_KEY='your-key'
```

### 입력 디렉토리 없음

```
Error: Input directory not found
→ create_dataset.py에서 input_dir 경로 확인
```

### 스킵 비율이 너무 높음

```
Skipped: 50 out of 100
→ skip_ambiguous=False 설정 고려
→ 또는 데이터 품질 확인
```

---

## 🚀 다음 단계

1. 샘플 테스트 실행

   ```bash
   python example_batch_process.py
   ```

2. API 키 설정 (선택사항)

   ```bash
   export OPENAI_API_KEY='your-key'
   ```

3. 전체 데이터셋 생성

   ```bash
   python create_dataset.py
   ```

4. 결과 확인
   ```bash
   cat ground_truth/geoqa3_dataset.json
   ```

---

**모든 기능이 정상 작동합니다! 질문이 있으면 언제든 물어보세요! 🎯**


