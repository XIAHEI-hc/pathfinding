# matrix_ring/matrix_ring.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 正确导入 colors 子模块
import logging
from typing import Tuple
plt.rc("font",family="Microsoft YaHei")
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MatrixRing:
    def __init__(self, outer_size: int, overall_size: int, inner_size: int):
        """
        初始化 MatrixRing 类。

        :param outer_size: 外圈大小
        :param overall_size: 整体矩阵大小
        :param inner_size: 内圈大小
        """
        self.outer_size = outer_size
        self.overall_size = overall_size
        self.inner_size = inner_size
        self.matrix = np.zeros((self.overall_size, self.overall_size), dtype=int)
        logging.info(f"初始化 MatrixRing 类: outer_size={outer_size}, overall_size={overall_size}, inner_size={inner_size}")

    def create_matrix(self):
        """
        创建矩阵，包含外圈、大圈和内圈。
        外圈与大圈之间为0，大圈与内圈之间为-1，内圈为1。
        """
        logging.info("开始创建矩阵")
        center = (self.overall_size // 2, self.overall_size // 2)
        logging.info(f"矩阵中心点: {center}")

        # 定义半径
        outer_radius = self.outer_size // 2
        big_ring_radius = outer_radius - 1  # 大圈半径
        inner_radius = self.inner_size // 2

        logging.info(f"外圈半径: {outer_radius}, 大圈半径: {big_ring_radius}, 内圈半径: {inner_radius}")

        for i in range(self.overall_size):
            for j in range(self.overall_size):
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                if distance < inner_radius:
                    self.matrix[i, j] = 1
                elif inner_radius <= distance < big_ring_radius:
                    self.matrix[i, j] = -1
                elif big_ring_radius <= distance < outer_radius:
                    self.matrix[i, j] = 0
                else:
                    self.matrix[i, j] = 0  # 外部区域保持为0
        logging.info("矩阵创建完成")
        return self.matrix

    def visualize(self):
        """
        可视化矩阵。
        最里面是绿色，最中间是红色，最外面是透明。
        每个格子标注其值。
        """
        logging.info("开始可视化矩阵")

        # 数据映射：
        # -1 -> 0 (red)
        # 1 -> 1 (green)
        # 0 -> masked (transparent)
        data_mapped = np.zeros_like(self.matrix, dtype=int)
        data_mapped[self.matrix == -1] = 0  # 红色
        data_mapped[self.matrix == 1] = 1  # 绿色

        # 创建 mask，mask 为 0 的地方
        mask = self.matrix == 0

        # 定义颜色映射，只包括红色和绿色
        cmap = mcolors.ListedColormap(['red', 'green'])
        cmap.set_bad(color=(0, 0, 0, 0))  # 设置被 mask 掩盖的区域为透明

        # 应用 mask
        masked_data = np.ma.masked_where(mask, data_mapped)

        fig, ax = plt.subplots(figsize=(8, 8))

        # 显示数据
        cax = ax.imshow(masked_data, cmap=cmap, interpolation='nearest')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='待测 DIE (1)'),
            Patch(facecolor='red', edgecolor='black', label='禁布区 (-1)'),
            Patch(facecolor='none', edgecolor='black', label='界外 (0)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        # 标注每个格子的值
        for i in range(self.overall_size):
            for j in range(self.overall_size):
                value = self.matrix[i, j]
                if value != 0:
                    text_color = 'white' if value == 1 or value == -1 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

        # 添加网格
        ax.set_xticks(np.arange(-0.5, self.overall_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.overall_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])

        plt.title('Matrix Ring Visualization')
        plt.show()
        logging.info("可视化完成")

    def read_file(self, file_path: str):
        """
        读取 Excel 或 CSV 文件并生成矩阵。

        :param file_path: 文件路径
        """
        logging.info(f"开始读取文件: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
            logging.info("CSV 文件读取成功")
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, header=None)
            logging.info("Excel 文件读取成功")
        else:
            logging.error("Unsupported file format. Please provide a CSV or Excel file.")
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        self.matrix = df.values
        logging.info("文件数据已转换为矩阵")
        return self.matrix

    def get_matrix(self) -> np.ndarray:
        """
        获取当前矩阵。

        :return: 矩阵
        """
        return self.matrix
