import os
import re
import matplotlib.pyplot as plt
import numpy as np
from geo_types import *


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

import commands as commands_module
from inspect import getmembers, isfunction
command_dict = dict(o for o in getmembers(commands_module) if isfunction(o[1]))

class Element:
    def __init__(self, label, element_dict):
        self.data = None
        self.label = label
        assert(label not in element_dict)
        element_dict[label] = self

    def drawable(self):
        return isinstance(self.data, (Point, Line, Angle, Polygon, Circle, Vector))
    
    def draw(self, ax, corners, show_labels=True):
        if not self.drawable():
            return
        
        # 객체 그리기
        self.data.draw(ax, corners)
        
        # 레이블 표시
        if show_labels and not self.label.startswith('_'):
            self._draw_label(ax, corners)
    
    def _draw_label(self, ax, corners):
        """객체 레이블을 그림에 표시"""
        offset = 3  # 레이블 오프셋
        
        if isinstance(self.data, Point):
            # 점 레이블: 점 위쪽에 표시
            ax.text(self.data.a[0], self.data.a[1] + offset, self.label,
                   fontsize=9, ha='center', va='bottom', color='blue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7))
        
        elif isinstance(self.data, Circle):
            # 원 레이블: 중심 근처에 표시
            ax.text(self.data.c[0] + offset, self.data.c[1] + offset, self.label,
                   fontsize=8, ha='left', va='bottom', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7))
        
        elif isinstance(self.data, (Line, Segment, Ray)):
            # 선/선분 레이블: 중간점에 표시
            endpoints = self.data.get_endpoints(corners)
            if endpoints is not None and len(endpoints) == 2:
                mid_x = (endpoints[0][0] + endpoints[1][0]) / 2
                mid_y = (endpoints[0][1] + endpoints[1][1]) / 2
                ax.text(mid_x, mid_y + offset, self.label,
                       fontsize=8, ha='center', va='bottom', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.7))
        
        # elif isinstance(self.data, Angle):
        #     # 각도 레이블: 꼭짓점 근처에 표시
        #     ax.text(self.data.p[0] + offset, self.data.p[1] + offset, self.label,
        #            fontsize=8, ha='left', va='bottom', color='purple',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
        #                     edgecolor='none', alpha=0.7))
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
        elif isinstance(self.data, Polygon): return command_module.area_P(self.data).x
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
        f = command_dict[name]
        output_data = f(*input_data)
        if not isinstance(output_data, (tuple, list)): output_data = (output_data,)
        assert(len(output_data) == len(self.output_elements))
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
    # 주석 제거 (# 이후의 내용 무시)
    if '#' in line:
        line = line[:line.index('#')]
    
    tokens = line.split()
    if len(tokens) == 0:  # 빈 줄 처리
        return None
    if tokens[0] == "const":
        assert(len(tokens) == 5)
        datatype = str_to_const_type[tokens[1]]
        value = float(tokens[2])
        assert(tokens[3] == "->")
        label = tokens[4]
        element = Element(label, element_dict)
        command = ConstCommand(datatype, value, element, line_number=line_number, original_line=line.strip())
        element.command = command
        return command
    else:
        command_name = tokens[0]
        assert(tokens[1] == ":")
        labels = [None if token == "_" else token for token in tokens[2:]]
        arrow_index = labels.index("->")
        input_labels = labels[:arrow_index]
        output_labels = labels[arrow_index+1:]
        
        # 숫자 리터럴, 삼각함수를 자동으로 element로 변환
        input_elements = []
        for label in input_labels:
            # 1. 먼저 삼각함수 표현인지 확인 (cos, sin, tan)
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
            
            # 2. 각도 표기 확인 (degree: °, radian: rad/r)
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
            
            # 3. 숫자인지 확인
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
                # 숫자가 아니면 기존 element 찾기
                input_elements.append(element_dict[label])
        
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
                command = parse_command(line, self.element_dict, line_number=line_num)
                if command is None:  # 빈 줄이거나 주석 등
                    continue
                if isinstance(command, ConstCommand): command.apply()
                elif isinstance(command, Command):
                    if command.name == "prove":
                        [inp] = command.input_elements
                        [out] = command.output_elements
                        if out is not None: del self.element_dict[out.label]
                        assert(self.to_prove is None)
                        self.to_prove = inp

                    else: self.nc_commands.append(command)

        # assert(self.to_prove is not None)
        self.elements = list(self.element_dict.values())

    def run_commands(self):
        for command in self.nc_commands:
            try:
                command.apply()
            except Exception as e:
                # Enhanced error message with line information
                error_msg = f"Error at line {command.line_number}: `{command.original_line}`\n"
                error_msg += f"Reason: {type(e).__name__}: {str(e)}"
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
    #datadir = "ggb-benchmark/true"
    datadir = "patrik"
    construction = Construction()
    for filename in os.listdir(datadir):
        if not filename.endswith(".txt"): continue
        construction.load(os.path.join(datadir, filename))
        construction.test()
