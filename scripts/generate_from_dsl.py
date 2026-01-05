#!/usr/bin/env python3
"""
DSL 파일로부터 기하학적 그림을 생성하는 스크립트

사용법:
    python generate_from_dsl.py <dsl_file> [output_image]

예시:
    python generate_from_dsl.py my_geometry.txt
    python generate_from_dsl.py my_geometry.txt output.png
"""

import sys
import os
import matplotlib.pyplot as plt
from src.core.random_constr import Construction

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def generate_from_dsl(dsl_file: str, output_file: str = None, display_size=(400, 300), show_display=True):
    """
    DSL 파일을 읽어서 기하학적 구조를 생성하고 이미지로 저장

    Args:
        dsl_file: DSL 코드가 담긴 텍스트 파일 경로
        output_file: 출력 이미지 파일 경로 (기본값: dsl_file과 같은 이름의 .png)
        display_size: 렌더링 크기 (width, height)
        show_display: 생성된 이미지를 화면에 표시할지 여부
    """
    # 파일 존재 확인
    if not os.path.exists(dsl_file):
        print(f"❌ 파일을 찾을 수 없습니다: {dsl_file}")
        return False

    # 출력 파일 이름 자동 생성
    if output_file is None:
        base_name = os.path.splitext(dsl_file)[0]
        output_file = f"{base_name}.png"

    print(f"📄 DSL 파일 읽기: {dsl_file}")

    try:
        # Construction 객체 생성
        construction = Construction(display_size=display_size)

        # DSL 파일 로드
        construction.load(dsl_file)
        print(f"✅ DSL 코드 로드 완료")

        # 기하학적 구조 생성
        print(f"🔨 기하학적 구조 생성 중...")
        construction.generate(require_theorem=False, max_attempts=1)
        print(f"✅ 구조 생성 완료")

        # 이미지 렌더링
        fig, ax = plt.subplots(figsize=(display_size[0]/100, display_size[1]/100), dpi=100)
        construction.render(ax)

        # 이미지 저장
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        print(f"💾 이미지 저장: {output_file}")

        # 화면에 표시
        if show_display:
            plt.show()
        else:
            plt.close(fig)

        return True

    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        return False
    except KeyError as e:
        print(f"❌ DSL 오류: 정의되지 않은 요소 또는 명령어 - {e}")
        return False
    except ValueError as e:
        print(f"❌ DSL 구문 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("사용법: python generate_from_dsl.py <dsl_file> [output_image] [--no-display]")
        print()
        print("예시:")
        print("  python generate_from_dsl.py my_triangle.txt")
        print("  python generate_from_dsl.py my_triangle.txt output.png")
        print("  python generate_from_dsl.py my_triangle.txt --no-display")
        print()
        print("DSL 예시 파일은 examples/ 디렉토리를 참고하세요:")
        print("  python generate_from_dsl.py examples/simple_circle.txt")
        sys.exit(1)

    dsl_file = sys.argv[1]

    # Parse arguments
    output_file = None
    show_display = True

    for arg in sys.argv[2:]:
        if arg == '--no-display':
            show_display = False
        elif not arg.startswith('--'):
            output_file = arg

    success = generate_from_dsl(dsl_file, output_file, show_display=show_display)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
