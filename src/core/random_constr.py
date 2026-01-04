import os
import re
import matplotlib.pyplot as plt
import numpy as np
from src.core.geo_types import *


def parse_trig_function(token: str) -> float:
    """
    Parse trigonometric function expressions like cos(30°), sin(45°), tan(60°).

    Supports:
    - cos(30°), sin(45°), tan(60°)  - degree input
    - cos(1.5708rad), sin(0.5rad), tan(0.785rad)  - radian input
    - cos(30), sin(45), tan(60)  - degree input (default)

    Returns:
        Calculated float value, or None if not a trig function
    """
    # Pattern: func(value[°|rad|r])
    trig_pattern = r'^(cos|sin|tan)\(([+-]?[\d.]+)(°|deg|rad|r)?\)$'
    match = re.match(trig_pattern, token, re.IGNORECASE)

    if not match:
        return None

    func_name = match.group(1).lower()
    value_str = match.group(2)
    unit = match.group(3)

    try:
        value = float(value_str)
    except ValueError:
        return None

    # Convert to radians if needed
    if unit in ('°', 'deg', None):
        # Default is degree
        value_rad = np.radians(value)
    else:
        # Already in radians
        value_rad = value

    # Calculate trig function
    if func_name == 'cos':
        return float(np.cos(value_rad))
    elif func_name == 'sin':
        return float(np.sin(value_rad))
    elif func_name == 'tan':
        return float(np.tan(value_rad))

    return None


def evaluate_math_expression(expr: str) -> float:
    """
    Evaluate mathematical expressions with trigonometric functions.

    Supports:
    - Arithmetic operations: +, -, *, /, (, )
    - Trigonometric functions: cos(angle), sin(angle), tan(angle)
    - Degree notation: ° (e.g., cos(100°))
    - Numbers: integers and floats

    IMPORTANT: Expressions must NOT contain spaces!
    - ✅ CORRECT: "(100+100*cos(115°))"
    - ❌ WRONG:   "(100 + 100*cos(115°))"

    Examples:
    - "100*cos(100°)" -> -17.36...
    - "100*sin(100°)" -> 98.48...
    - "50+30*cos(45°)" -> 71.21...
    - "2*sin(30°)+3*cos(60°)" -> 2.5
    - "(100+100*cos(115°))" -> 57.74...

    Returns:
        Calculated float value, or None if expression is invalid
    """
    try:
        # Replace trigonometric functions with their calculated values
        # Pattern: cos(value°), sin(value°), tan(value°)
        def replace_trig(match):
            func_name = match.group(1).lower()
            value_str = match.group(2)
            unit = match.group(3)

            value = float(value_str)

            # Convert to radians if needed
            if unit in ('°', 'deg', None):
                value_rad = np.radians(value)
            else:
                value_rad = value

            # Calculate trig function
            if func_name == 'cos':
                result = np.cos(value_rad)
            elif func_name == 'sin':
                result = np.sin(value_rad)
            elif func_name == 'tan':
                result = np.tan(value_rad)
            else:
                return match.group(0)  # Return original if unknown function

            return str(float(result))

        # Replace all trig functions in the expression
        trig_pattern = r'(cos|sin|tan)\(([+-]?[\d.]+)(°|deg|rad|r)?\)'
        processed_expr = re.sub(trig_pattern, replace_trig, expr, flags=re.IGNORECASE)

        # Remove any remaining degree symbols (in case of standalone numbers like "100°")
        processed_expr = processed_expr.replace('°', '')

        # Evaluate the expression safely
        # Only allow numbers, basic operators, parentheses, and scientific notation (e/E)
        # NOTE: Spaces are NOT allowed - expressions must be written without spaces
        allowed_chars = set('0123456789.+-*/()eE')
        if not all(c in allowed_chars for c in processed_expr):
            return None

        # Use eval with restricted globals/locals for safety
        result = eval(processed_expr, {"__builtins__": {}}, {})
        return float(result)

    except Exception:
        return None

type_to_shortcut = {
    int       : 'i',
    float     : 'i',  # float는 int와 동일하게 처리 (commands에서 float()로 변환)
    Boolean   : 'b',
    Measure   : 'm',
    Point     : 'p',
    Polygon   : 'P',
    Circle    : 'c',
    Arc       : 'C',
    Line      : 'l',
    Ray       : 'r',
    Segment   : 's',
    Angle     : 'a',
    AngleSize : 'A',
    Vector    : 'v',
}

def command_types_name(name, params):
    return "{}_{}".format(name, ''.join(type_to_shortcut[type(x)] for x in params))

from src.core import commands as commands_module
from inspect import getmembers, isfunction
command_dict = dict(o for o in getmembers(commands_module) if isfunction(o[1]))

class Element:
    def __init__(self, label, element_dict):
        self.data = None
        self.label = label
        if label in element_dict:
            raise ValueError(f"Duplicate label '{label}'. This element name is already defined.")
        element_dict[label] = self

    def drawable(self):
        return isinstance(self.data, (Point, Line, Polygon, Circle, Vector))
    
    def draw(self, ax, corners, show_labels=True):
        if not self.drawable():
            return
        
        # 객체 그리기
        self.data.draw(ax, corners)
        
        # 레이블 표시
        if show_labels and not self.label.startswith('_'):
            self._draw_label(ax, corners)
    
    def _draw_label(self, ax, corners):
        """객체 레이블을 그림에 표시 (점에만 표시)"""
        offset = 5  # 레이블 오프셋

        if isinstance(self.data, Point):
            # 점 레이블: 점 위쪽에 표시
            ax.text(self.data.a[0], self.data.a[1] + offset, self.label,
                   fontsize=12, ha='center', va='bottom', color='black',
                   fontfamily='serif', fontweight='normal')
    def important_points(self):
        if self.drawable(): return self.data.important_points()
        else: return []

    def has_value(self):
        return isinstance(self.data, (Measure, Boolean, AngleSize, Angle, Segment, Polygon))
    def value(self):
        if isinstance(self.data, (Measure, AngleSize)): return self.data.x
        elif isinstance(self.data, Boolean): return float(self.data.b)
        elif isinstance(self.data, Angle): return self.data.angle
        elif isinstance(self.data, Segment): return self.data.length
        elif isinstance(self.data, Polygon): return commands_module.area_P(self.data).x
        else: return None

class Command:
    def __init__(self, command_name, input_elements, output_elements, line_number=None, original_line=None):
        self.name = command_name
        self.input_elements = input_elements
        self.output_elements = output_elements
        self.line_number = line_number
        self.original_line = original_line

    def apply(self):
        input_data = [x.data for x in self.input_elements]
        name = command_types_name(self.name, input_data)
        if name not in command_dict: name = self.name

        # Check if command exists
        if name not in command_dict:
            input_types = ', '.join([type(x).__name__ for x in input_data])
            raise KeyError(f"Unknown command '{self.name}' with input types ({input_types}). "
                         f"This command may not exist or may not accept these input types.")

        f = command_dict[name]
        output_data = f(*input_data)
        if not isinstance(output_data, (tuple, list)): output_data = (output_data,)

        # Check output count
        if len(output_data) != len(self.output_elements):
            raise AssertionError(f"Command '{self.name}' returned {len(output_data)} output(s), "
                               f"but {len(self.output_elements)} output(s) were expected. "
                               f"Please check the command documentation.")

        for x,o in zip(output_data, self.output_elements):
            if o is not None: o.data = x

    def __str__(self):
        inputs_str = ' '.join([x.label for x in self.inputs])
        outputs_str = ' '.join([x.label if x is not None else "_" for x in self.outputs])
        return "{} : {} -> {}".format(
            self.name, inputs_str, outputs_str
        )

const_type_to_str = {
    int : "int",
    AngleSize : "AngleSize",
    Measure : "Measure",
}
str_to_const_type = dict((s,t) for (t,s) in const_type_to_str.items())

class ConstCommand:
    def __init__(self, datatype, value, element, line_number=None, original_line=None):
        self.datatype = datatype
        self.value = value
        self.element = element
        self.line_number = line_number
        self.original_line = original_line

    def apply(self):
        self.element.data = self.datatype(self.value)

    def __str__(self):
        datatype_str = const_type_to_str[self.datatype]
        return "const {} {} -> {}".format(datatype_str, self.value, self.label)

def parse_command(line, element_dict, line_number=None):
    # Discard comments
    if '#' in line:
        line = line[:line.index('#')]
    
    tokens = line.split()
    if len(tokens) == 0:  # Handle empty lines
        return None
    if tokens[0] == "const":
        if len(tokens) != 5:
            raise ValueError(f"const command requires exactly 5 tokens, got {len(tokens)}. "
                           f"Format: const <type> <value> -> <label>")
        try:
            datatype = str_to_const_type[tokens[1]]
        except KeyError:
            raise ValueError(f"Invalid const type '{tokens[1]}'. Valid types: {', '.join(const_type_to_str.values())}")

        try:
            value = float(tokens[2])
        except ValueError:
            raise ValueError(f"Invalid numeric value '{tokens[2]}' for const command")

        if tokens[3] != "->":
            raise ValueError(f"Expected '->' at position 4, got '{tokens[3]}'")

        label = tokens[4]
        element = Element(label, element_dict)
        command = ConstCommand(datatype, value, element, line_number=line_number, original_line=line.strip())
        element.command = command
        return command
    else:
        command_name = tokens[0]
        if len(tokens) < 2 or tokens[1] != ":":
            raise ValueError(f"Invalid command format. Expected '<command> : ...', got '{' '.join(tokens[:2])}'")

        labels = [None if token == "_" else token for token in tokens[2:]]
        try:
            arrow_index = labels.index("->")
        except ValueError:
            raise ValueError(f"Missing '->' in command. Format: {command_name} : <inputs> -> <outputs>")

        input_labels = labels[:arrow_index]
        output_labels = labels[arrow_index+1:]

        if len(input_labels) == 0:
            raise ValueError(f"No inputs provided for command '{command_name}'")
        if len(output_labels) == 0:
            raise ValueError(f"No outputs provided for command '{command_name}'")
        
        # 숫자 리터럴, 삼각함수, 수식을 자동으로 element로 변환
        # 우선순위: 1) 기존 element, 2) 수식, 3) 삼각함수, 4) 숫자
        input_elements = []
        for label in input_labels:
            # 1. 먼저 기존 element_dict에서 찾기 시도 (가장 우선순위 높음)
            if label in element_dict:
                input_elements.append(element_dict[label])
                continue

            # 2. 수식 평가 시도 (산술 연산이나 삼각함수가 포함된 경우)
            # 수식일 가능성이 있는지 확인 (연산자나 삼각함수 포함)
            if any(op in label for op in ['*', '+', '-', '/', '(', ')']) or \
               any(func in label.lower() for func in ['cos', 'sin', 'tan']):
                expr_value = evaluate_math_expression(label)
                if expr_value is not None:
                    # 수식 계산 성공
                    auto_label = f"_auto_{len(element_dict)}"
                    while auto_label in element_dict:
                        auto_label = f"_auto_{len(element_dict)}_{np.random.randint(10000)}"

                    element = Element(auto_label, element_dict)
                    element.data = float(expr_value)
                    element.command = None
                    input_elements.append(element)
                    continue

            # 3. 단순 삼각함수 표현인지 확인 (이전 방식과의 호환성)
            trig_value = parse_trig_function(label)
            if trig_value is not None:
                # 삼각함수 값을 float로 저장
                auto_label = f"_auto_{len(element_dict)}"
                while auto_label in element_dict:
                    auto_label = f"_auto_{len(element_dict)}_{np.random.randint(10000)}"

                element = Element(auto_label, element_dict)
                element.data = trig_value
                element.command = None
                input_elements.append(element)
                continue

            # 4. 각도 표기 확인 (degree: °, radian: rad/r)
            is_degree = label.endswith('°')
            is_radian = label.endswith('rad') or label.endswith('r')

            # 각도 표기 제거
            clean_label = label
            if is_degree:
                clean_label = label[:-1]  # ° 제거
            elif is_radian:
                if label.endswith('rad'):
                    clean_label = label[:-3]  # rad 제거
                else:
                    clean_label = label[:-1]  # r 제거

            # 5. 숫자인지 확인
            try:
                value = float(clean_label)
                # 자동 레이블 생성 (충돌 방지)
                auto_label = f"_auto_{len(element_dict)}"
                while auto_label in element_dict:
                    auto_label = f"_auto_{len(element_dict)}_{np.random.randint(10000)}"

                # 임시 element 생성
                element = Element(auto_label, element_dict)

                # 각도 타입에 따라 처리
                if is_radian:
                    # radian → AngleSize 객체로 저장
                    element.data = AngleSize(value)
                elif is_degree:
                    # degree → radian으로 변환하여 AngleSize 객체로 저장
                    element.data = AngleSize(np.radians(value))
                else:
                    # 일반 숫자: 소수점이 있으면 float, 없으면 int
                    if '.' in clean_label:
                        element.data = float(value)
                    else:
                        element.data = int(value)

                # 더미 command 생성 (추적용)
                element.command = None
                input_elements.append(element)
            except ValueError:
                # 숫자도 아니고 기존 element도 아니면 에러
                raise KeyError(f"Unknown element or invalid expression: '{label}'")
        
        def element_or_none(label):
            if label is None: return None
            else: return Element(label, element_dict)
        output_elements = list(map(element_or_none, output_labels))
        command = Command(command_name, input_elements, output_elements, line_number=line_number, original_line=line.strip())
        for el in output_elements:
            if el is not None: el.command = command
        return command

class Construction:
    def __init__(self, display_size = (100,100), min_border = 0.1, max_border = 0.25):
        self.corners = np.array(((0,0), display_size))
        self.min_border = min_border
        self.max_border = max_border
        self.nc_commands = []
        self.to_prove = None
        self.element_dict = dict()
        self.elements = []
        self.scale_factor = 1.0  # Track scaling applied by fit_to_window()

    def render(self, ax, elements = None, show_labels=True): # default: render all elements
        if elements is None: elements = self.elements

        # Clear the axes
        ax.clear()
        
        # Set aspect ratio to equal and remove axes
        ax.set_aspect('equal')
        ax.set_xlim(self.corners[0][0], self.corners[1][0])
        ax.set_ylim(self.corners[0][1], self.corners[1][1])
        ax.axis('off')
        
        # Draw all elements
        for el in elements:
            el.draw(ax, self.corners, show_labels=show_labels)

    def render_to_numpy(self, elements = None):
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(self.corners[1][0]/100, self.corners[1][1]/100), dpi=100)
        self.render(ax, elements)

        # Convert to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert to grayscale
        data = np.mean(data, axis=2) / 255
        return data

    def load(self, filename):
        self.nc_commands = []
        self.to_prove = None
        self.element_dict = dict()
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    command = parse_command(line, self.element_dict, line_number=line_num)
                except Exception as e:
                    # Wrap parse errors with line information
                    original_line = line.strip()
                    error_msg = f"Parse error at line {line_num}: `{original_line}`\n\n"
                    error_msg += f"❌ {type(e).__name__}: {str(e)}\n"
                    error_msg += f"\n💡 TIP: Check the DSL syntax for this line."
                    raise RuntimeError(error_msg) from e

                if command is None:  # 빈 줄이거나 주석 등
                    continue
                if isinstance(command, ConstCommand): command.apply()
                elif isinstance(command, Command):
                    if command.name == "prove":
                        [inp] = command.input_elements
                        [out] = command.output_elements
                        if out is not None: del self.element_dict[out.label]
                        if self.to_prove is not None:
                            raise RuntimeError(f"Multiple 'prove' statements found. Only one is allowed.")
                        self.to_prove = inp

                    else: self.nc_commands.append(command)

        # assert(self.to_prove is not None)
        self.elements = list(self.element_dict.values())

    def run_commands(self):
        for command in self.nc_commands:
            try:
                command.apply()
            except Exception as e:
                # Enhanced error message with line information and specific diagnostics
                error_msg = f"Error at line {command.line_number}: `{command.original_line}`\n\n"

                # Determine specific error type and provide helpful message
                error_type = type(e).__name__
                error_str = str(e)

                # Provide detailed error explanation
                if error_type == "KeyError":
                    # Check if this is an unknown command error
                    if "Unknown command" in error_str:
                        cleaned_error = error_str.strip("'\"")
                        error_msg += f"❌ {cleaned_error}\n\n"
                        error_msg += f"Suggestions:\n"
                        error_msg += f"   - Check if the command name is spelled correctly\n"
                        error_msg += f"   - Verify that the input types are correct for this command\n"
                        error_msg += f"   - Some commands have type-specific variants (e.g., intersect_ll for lines)\n"
                    else:
                        # Undefined element reference
                        undefined_elem = error_str.strip("'\"")
                        error_msg += f"❌ UNDEFINED ELEMENT: '{undefined_elem}'\n"
                        error_msg += f"   The element '{undefined_elem}' is used but not defined.\n"
                        error_msg += f"   Make sure to define it before using it in a command.\n\n"
                        error_msg += f"Available elements: {', '.join(sorted(self.element_dict.keys())[:10])}"
                        if len(self.element_dict) > 10:
                            error_msg += f" ... and {len(self.element_dict) - 10} more"

                elif error_type == "ValueError":
                    # Geometric degenerate cases or invalid values
                    error_msg += f"❌ GEOMETRIC ERROR\n"
                    error_msg += f"   {error_str}\n\n"
                    error_msg += f"💡 TIP: Check that your geometric construction is valid.\n"
                    error_msg += f"   Common issues:\n"
                    error_msg += f"   - Trying to create a line/segment through identical points\n"
                    error_msg += f"   - Trying to intersect parallel lines\n"
                    error_msg += f"   - Intersection point outside ray/segment bounds\n"
                    error_msg += f"   - Circle with zero or negative radius\n"

                elif error_type == "AssertionError":
                    # Now only legitimate output count mismatches should reach here
                    if hasattr(command, 'output_elements'):
                        error_msg += f"❌ OUTPUT COUNT MISMATCH\n"
                        error_msg += f"   {error_str}\n"
                        error_msg += f"   Command '{command.name}' returned wrong number of outputs.\n\n"
                        error_msg += f"   Common causes:\n"
                        error_msg += f"   - intersect command may return 1 or 2 points depending on geometry\n"
                        error_msg += f"   - Command signature may be incorrect\n"
                    else:
                        error_msg += f"❌ ASSERTION ERROR\n"
                        error_msg += f"   {error_str}\n"
                        error_msg += f"   Invalid command format or constraint violation.\n"

                elif "Unknown element or invalid expression" in error_str:
                    error_msg += f"❌ {error_str}\n"
                    error_msg += f"   This could be:\n"
                    error_msg += f"   - A typo in an element name\n"
                    error_msg += f"   - An invalid mathematical expression\n"
                    error_msg += f"   - Missing element definition\n"

                else:
                    # Generic error
                    error_msg += f"❌ {error_type}: {error_str if error_str else '(no details)'}\n"

                # Add context
                error_msg += f"\n💡 TIP: Check the DSL syntax and ensure all elements are defined before use."

                raise RuntimeError(error_msg) from e

    def generate(self, require_theorem = True, max_attempts = 100): # max_attempts = 0 -> inf
        while True:
            try:
                self.run_commands()
            except:
                max_attempts -= 1
                if max_attempts == 0: raise
                continue
            if require_theorem and not self.to_prove.data.b: continue
            break

        self.fit_to_window()

    def fit_to_window(self):
        important_points = []
        for el in self.elements: important_points += el.important_points()
        if len(important_points) == 0: return
        src_corners = np.stack([
            np.min(important_points, axis = 0),
            np.max(important_points, axis = 0),
        ])
        src_size = np.maximum(0.01, src_corners[1] - src_corners[0])

        dest_size = self.corners[1] - self.corners[0]
        dest_corners_shift = np.random.random(size = [2,2])
        dest_corners_shift *= self.max_border - self.min_border
        dest_corners_shift += self.min_border
        dest_corners_shift *= np.array((1,-1)).reshape((2,1)) * dest_size
        dest_corners = self.corners + dest_corners_shift
        dest_size = dest_corners[1] - dest_corners[0]

        scale = np.min(dest_size / src_size)
        self.scale_factor = scale  # Store scale factor for validation
        src_corners *= scale
        shift = np.average(dest_corners, axis = 0) - np.average(src_corners, axis = 0)
        for el in self.elements:
            if isinstance(el.data, (int, float)): continue
            el.data.scale(scale)
            el.data.translate(shift)

        important_points = []
        for el in self.elements: important_points += el.important_points()
        corners = np.stack([
            np.min(important_points, axis = 0),
            np.max(important_points, axis = 0),
        ])

    def test(self, num_tests = 100):
        constr_fail, check_fail, success = 0, 0, 0
        for _ in range(num_tests):
            try:
                self.run_commands()
                if self.to_prove.data.b: success += 1
                else: check_fail += 1
            except:
                constr_fail += 1

        constr_fail, check_fail, success = [
            100*x / num_tests
            for x in (constr_fail, check_fail, success)
        ]
        print("{:.2f}% failed constructions, {:.2f}% false, {:.2f}% true".format(
            constr_fail, check_fail, success
        ))

if __name__ == "__main__":
    from src.utils import get_project_root
    #datadir = "ggb-benchmark/true"
    datadir = get_project_root() / "patrik"
    construction = Construction()
    for filename in os.listdir(datadir):
        if not filename.endswith(".txt"): continue
        construction.load(os.path.join(datadir, filename))
        construction.test()











