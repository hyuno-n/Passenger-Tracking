from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSpacerItem, QSizePolicy, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import os
import json

class SeatButton(QPushButton):
    def __init__(self, seat_id, parent=None):
        super().__init__(seat_id, parent)
        self.seat_id = seat_id
        self.is_selected = False
        self.update_style()

        self.clicked.connect(self.toggle_state)

    def toggle_state(self):
        self.is_selected = not self.is_selected
        self.update_style()

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("background-color: lightgreen;")
        else:
            self.setStyleSheet("background-color: lightgray;")

    def reset(self):
        self.is_selected = False
        self.update_style()

class SeatEvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("좌석 점유 인지 평가 도구")
        self.resize(1000, 900)

        self.current_frame = 0
        self.total_frames = 100  # 예시

        self.user_results = {}  # 사용자 판단 결과 저장용

        self.image_folder = None
        self.image_files = []
        self.all_results = {}  # frame → seat_id → bool

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.select_folder_btn = QPushButton("📂 폴더 선택")
        self.select_folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_btn)

        # 어안 이미지 영역
        self.image_label = QLabel("Fisheye Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap("test.jpg").scaledToWidth(600))
        layout.addWidget(self.image_label)

        # matplotlib 좌석 배치도 (좌우 반전 포함)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.draw_seat_map()

        # 좌석 버튼 (배치도 반영, 좌우 반전)
        self.seat_layout = QGridLayout()
        self.seat_buttons = {}

        seat_rows = [
            ["S7", "S6", "S5", "S4", "S3", "S2", "S1"],
            ["",   "",   "S12","S11","S10","S9", "S8"],
            ["",   "",   "S15","S14","S13","",""]
        ]

        for row_idx, row in enumerate(seat_rows):
            actual_row = row_idx
            if row_idx == 2:  # 셋째 줄(세 번째 줄)을 row 3에 배치 (빈 줄 확보)
                actual_row += 1  # row 2 → row 3

            for col_idx, seat_id in enumerate(row):
                if seat_id:
                    btn = SeatButton(seat_id)
                    self.seat_buttons[seat_id] = btn
                    self.user_results[seat_id] = False
                    self.seat_layout.addWidget(btn, actual_row, col_idx)

        # row 2에 높이 30짜리 빈 공간 추가
        spacer = QSpacerItem(0, 50, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.seat_layout.addItem(spacer, 2, 0, 1, 7)  # 1행 7열 차지
        layout.addLayout(self.seat_layout)

        # 프레임 이동 + 초기화 버튼
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev Frame")
        self.save_btn = QPushButton("Save")
        self.reset_btn = QPushButton("Reset")
        self.next_btn = QPushButton("Next Frame ▶")

        self.prev_btn.clicked.connect(self.prev_frame)
        self.save_btn.clicked.connect(self.save_all_results)
        self.reset_btn.clicked.connect(self.reset_selection)
        self.next_btn.clicked.connect(self.next_frame)

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def draw_seat_map(self):
        self.ax.clear()
        self.ax.set_title("seating chart")
        self.ax.set_xlim(0, 7)
        self.ax.set_ylim(0, 4)
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        seat_positions = {
            "S7": (0, 0), "S6": (1, 0), "S5": (2, 0), "S4": (3, 0),
            "S3": (4, 0), "S2": (5, 0), "S1": (6, 0),
            "S12": (2, 1), "S11": (3, 1), "S10": (4, 1),
            "S9": (5, 1), "S8": (6, 1),
            "S15": (2, 3), "S14": (3, 3), "S13": (4, 3)
        }

        for seat_id, (x, y) in seat_positions.items():
            self.ax.add_patch(plt.Rectangle((x, y), 1, 1, color='orange', alpha=0.5))
            self.ax.text(x + 0.3, y + 0.6, seat_id, fontsize=8)

        self.canvas.draw()

    def collect_user_input(self):
        current_file = self.image_files[self.current_frame] if self.image_files else f"frame_{self.current_frame:04d}.jpg"
        seat_states = {seat_id: btn.is_selected for seat_id, btn in self.seat_buttons.items()}
        self.all_results[current_file] = seat_states
        print(f"[{current_file}] 사용자 판단 결과: {seat_states}")

    def reset_selection(self):
        for btn in self.seat_buttons.values():
            btn.reset()
        print("좌석 선택 초기화 완료")

    def next_frame(self):
        self.collect_user_input()
        if self.current_frame < len(self.image_files) - 1:
            self.current_frame += 1
            self.load_frame_image()

    def prev_frame(self):
        self.collect_user_input()
        if self.current_frame > 0:
            self.current_frame -= 1
            self.load_frame_image()


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder:
            self.image_folder = folder
            self.image_files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            self.current_frame = 0
            print(f"선택된 폴더: {folder}")
            print(f"{len(self.image_files)}개의 이미지 로딩됨.")
            self.load_frame_image()

    def load_frame_image(self):
        if self.image_folder and self.image_files:
            frame_file = self.image_files[self.current_frame]
            path = os.path.join(self.image_folder, frame_file)
            self.image_label.setPixmap(QPixmap(path).scaledToWidth(600))

    def save_all_results(self):
        if not self.image_folder:
            print("폴더를 먼저 선택하세요.")
            return
        save_path = os.path.join(self.image_folder, "user_labels.json")
        with open(save_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"결과가 {save_path}에 저장되었습니다.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeatEvaluationApp()
    window.show()
    sys.exit(app.exec_())
