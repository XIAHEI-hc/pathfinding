# grid_selector/grid_selector.py

import tkinter as tk
from tkinter import messagebox
import numpy as np
import logging
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GridSelector:
    def __init__(self, width: int, height: int, cell_size: int = 20):
        """
        初始化 GridSelector 类。

        :param width: 网格的列数
        :param height: 网格的行数
        :param cell_size: 每个格子的大小（像素）
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.matrix = np.zeros((self.height, self.width), dtype=int)
        self.root = tk.Tk()
        self.root.title("Grid Selector")
        self.canvas = tk.Canvas(self.root, width=self.width * self.cell_size,
                                height=self.height * self.cell_size, bg='white')
        self.canvas.pack()
        self.confirm_button = tk.Button(self.root, text="确认选择", command=self.confirm_selection)
        self.confirm_button.pack(pady=10)
        self.rectangles = {}
        self.start_x = None
        self.start_y = None
        self.dragging = False
        self.processed_cells = set()

        logging.info(f"初始化 GridSelector 类: width={width}, height={height}, cell_size={cell_size}")

        self.draw_grid()
        self.bind_events()

    def draw_grid(self):
        """绘制初始网格，所有格子为灰色（0）"""
        logging.info("开始绘制网格")
        for i in range(self.height):
            for j in range(self.width):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill='gray', outline='black')
                self.rectangles[(i, j)] = rect
        logging.info("网格绘制完成")

    def bind_events(self):
        """绑定鼠标事件"""
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        """处理单击事件"""
        row, col = self.get_cell(event.x, event.y)
        if row is not None and col is not None:
            self.toggle_cell(row, col)
            self.start_x, self.start_y = event.x, event.y
            self.dragging = True
            logging.info(f"点击格子: ({row}, {col})")

    def on_drag(self, event):
        """处理拖动事件，选择多个格子"""
        if not self.dragging:
            return
        current_x, current_y = event.x, event.y
        row_start, col_start = self.get_cell(self.start_x, self.start_y)
        row_end, col_end = self.get_cell(current_x, current_y)
        if row_start is None or col_start is None or row_end is None or col_end is None:
            return
        row_min, row_max = sorted([row_start, row_end])
        col_min, col_max = sorted([col_start, col_end])
        logging.info(f"拖动选择区域: 行 {row_min} 到 {row_max}, 列 {col_min} 到 {col_max}")
        for i in range(row_min, row_max + 1):
            for j in range(col_min, col_max + 1):
                if (i, j) not in self.processed_cells:
                    self.toggle_cell(i, j)
                    self.processed_cells.add((i, j))

    def on_release(self, event):
        """处理鼠标释放事件，完成拖动选择"""
        self.dragging = False
        self.processed_cells.clear()
        logging.info("完成拖动选择")

    def get_cell(self, x: int, y: int) -> Tuple[int, int]:
        """根据鼠标坐标获取对应的格子行列"""
        col = x // self.cell_size
        row = y // self.cell_size
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None, None

    def toggle_cell(self, row: int, col: int):
        """切换格子的状态（0 <-> 1）"""
        current_state = self.matrix[row, col]
        new_state = 1 if current_state == 0 else 0
        self.set_cell_state(row, col, state=new_state)
        logging.info(f"切换格子 ({row}, {col}) 状态为 {new_state}")

    def set_cell_state(self, row: int, col: int, state: int = None, toggle: bool = True):
        """
        设置格子的状态。

        :param row: 行号
        :param col: 列号
        :param state: 要设置的状态（0 或 1），如果为 None 则根据 toggle 切换
        :param toggle: 是否切换状态
        """
        if toggle:
            state = 1 if self.matrix[row, col] == 0 else 0
        if state not in [0, 1]:
            return
        self.matrix[row, col] = state
        color = 'yellow' if state == 1 else 'gray'
        self.canvas.itemconfig(self.rectangles[(row, col)], fill=color)
        logging.info(f"设置格子 ({row}, {col}) 颜色为 {'yellow' if state == 1 else 'gray'}")

    def confirm_selection(self):
        """确认选择并关闭界面"""
        logging.info("用户确认选择，生成矩阵")
        self.root.destroy()

    def show(self) -> np.ndarray:
        """显示界面并返回矩阵"""
        self.root.mainloop()
        logging.info("界面关闭，返回矩阵")
        return self.matrix

    def trim_matrix(self) -> np.ndarray:
        """
        截取包含所有1的最小矩阵，去掉最外圈的所有0。

        :return: 截取后的矩阵
        """
        logging.info("开始截取包含所有1的最小矩阵")
        if not np.any(self.matrix == 1):
            logging.warning("矩阵中没有1，返回空矩阵")
            return np.array([[]], dtype=int)

        rows_with_one = np.any(self.matrix == 1, axis=1)
        cols_with_one = np.any(self.matrix == 1, axis=0)

        top, bottom = np.where(rows_with_one)[0][[0, -1]]
        left, right = np.where(cols_with_one)[0][[0, -1]]

        logging.info(f"最上边含1的行: {top}")
        logging.info(f"最下边含1的行: {bottom}")
        logging.info(f"最左边含1的列: {left}")
        logging.info(f"最右边含1的列: {right}")

        trimmed_matrix = self.matrix[top:bottom+1, left:right+1]
        logging.info(f"截取后的矩阵形状: {trimmed_matrix.shape}")
        return trimmed_matrix
