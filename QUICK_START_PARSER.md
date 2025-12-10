# Quick Start Guide - Problem Parser

## 🚀 빠른 시작 (Quick Start)

### 1단계: OpenAI API 키 설정

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2단계: 데이터셋 생성

```bash
# create_dataset.py 파일을 열어서 경로 수정
# - input_dir: JSON 파일이 있는 디렉토리
# - output_file: 출력 파일 경로

# 실행
python create_dataset.py
```

## 📖 주요 함수 (Main Functions)

### 개별 문제 파싱

```python
from problem_parser import ProblemParser, create_openai_api_function
import os

# 파서 초기화
api_key = os.getenv("OPENAI_API_KEY")
llm_func = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
parser = ProblemParser(llm_api_function=llm_func)

# 문제 파싱
result = parser.parse_problem(
    problem_text="여기에 문제 텍스트",
    problem_id="1",
    skip_ambiguous=True,  # ∠1, ∠2 등 건너뛰기
    clean_text=True       # 텍스트 정리
)

# 결과 확인
if result:
    print(result['category'])      # 문제 분류
    print(result['difficulty'])    # 난이도 (1-5)
    print(result['cleaned_text'])  # 정리된 텍스트
else:
    print("문제가 건너뛰어졌습니다")
```

### 배치 처리

```python
# 디렉토리의 모든 파일 처리
stats = parser.batch_parse_directory(
    input_dir="data-5/GeoQA3/json",
    output_file="output.json",
    skip_ambiguous=True,
    clean_text=True
)

print(f"성공: {stats['parsed']}, 건너뜀: {stats['skipped']}")
```

## ✨ 주요 기능

| 기능             | 메서드                       | 설명                   |
| ---------------- | ---------------------------- | ---------------------- |
| 모호한 참조 감지 | `has_ambiguous_references()` | ∠1, ∠2 등 감지         |
| 텍스트 정리      | `clean_problem_text()`       | "如图所示", 질문 제거  |
| 문제 분류        | `classify_problem()`         | 10가지 카테고리로 분류 |
| 난이도 평가      | `rate_difficulty()`          | 1-5 난이도 평가        |
| 전체 파싱        | `parse_problem()`            | 모든 기능 포함         |
| 배치 처리        | `batch_parse_directory()`    | 디렉토리 전체 처리     |

## 📊 출력 데이터 구조

```json
{
  "id": "1",
  "original_text": "원본 텍스트",
  "cleaned_text": "정리된 텍스트",
  "category": "Triangle Properties & Constructions",
  "difficulty": 3,
  "required_objects": {
    "points": ["A", "B", "C"],
    "segments": [["A", "B"]],
    "lines": [],
    "circles": [],
    "polygons": [["A", "B", "C"]]
  },
  "verification_conditions": [{ "type": "angle_value", "points": [["A", "B", "C"]], "value": 90 }]
}
```

## 🎯 분류 카테고리

1. Basic Constructions
2. Circle Properties & Constructions
3. Geometric Transformations
4. Triangle Properties & Constructions
5. Applications of Geometric Theorems
6. Polygon Properties & Constructions
7. Measurement & Ratios
8. Locus Constructions
9. Angle Relationships
10. Similarity & Congruence

## 📝 난이도 레벨

- **1**: 매우 쉬움 (기본 도형)
- **2**: 쉬움 (단순 구성)
- **3**: 보통 (중간 복잡도)
- **4**: 어려움 (복잡한 구성)
- **5**: 매우 어려움 (매우 복잡)

## 🧪 테스트

```bash
# 기능 테스트
python test_parser_features.py

# 예제 실행
python problem_parser.py
```

## ⚠️ 건너뛰어지는 문제

- ✗ `∠1=30°, ∠2=45°` (번호가 매겨진 각도)
- ✓ `∠ABC=30°, ∠BCD=45°` (이름이 있는 각도)

## 💡 팁

1. **API 키 필수**: 분류와 난이도 평가에는 OpenAI API 키가 필요합니다
2. **배치 처리 권장**: 많은 파일은 `batch_parse_directory()` 사용
3. **출력 확인**: 건너뛴 파일 수를 확인하여 데이터 품질 점검
4. **카테고리 수정 가능**: `ProblemParser.PROBLEM_CATEGORIES`에서 카테고리 변경 가능

## 📚 더 보기

- 상세 가이드: `PROBLEM_PARSER_GUIDE.md`
- 업데이트 요약: `PROBLEM_PARSER_UPDATE_KR.md`




