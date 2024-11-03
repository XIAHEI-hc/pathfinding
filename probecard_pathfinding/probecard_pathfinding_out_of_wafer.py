import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.colors as mcolors
import logging
from DIE_map.DIE_map import MatrixRing
from probercard_setting.probecard_UI import GridSelector
from probecard_pathfinding.uils import pad_matrix_with_zeros

class FloorPlanner:
    def __init__(self, floor_plan: np.ndarray, tile: np.ndarray):
        """
        初始化 FloorPlanner 类。

        :param floor_plan: 地板矩阵，1 表示可铺设区域，0 表示不可铺设区域。
        :param tile: 瓷砖矩阵，1 表示瓷砖的占用区域，0 表示空白区域。
        """
        self.floor_plan = floor_plan.copy()
        self.tile = tile
        self.tiled_floor = np.zeros_like(floor_plan)
        self.tile_positions = []
        self.logs = []
        self.remove_logs = []
        self.frames = []
        self.tiles_remaining = set()

    def place_tiles(self):
        """
        在地板上铺设瓷砖，记录每次铺设的位置和日志。
        """
        floor_rows, floor_cols = self.floor_plan.shape
        tile_rows, tile_cols = self.tile.shape
        floor_with_tiles = self.floor_plan.copy()

        for i in range(floor_rows - tile_rows + 1):
            for j in range(floor_cols - tile_cols + 1):
                floor_region = floor_with_tiles[i:i + tile_rows, j:j + tile_cols]

                print("瓷砖矩阵:")
                print(self.tile)
                print("地板区域:")
                print(floor_region)

                # 1. 检查是否有瓷砖1对应地板-1
                invalid_overlap = np.any((self.tile == 1) & (floor_region == -1))
                if invalid_overlap:
                    print("有瓷砖1对应地板-1，无法贴上瓷砖。")
                    self.logs.append(f"无法在位置 ({i}, {j}) 放置瓷砖：瓷砖1对应地板-1")
                    continue  # 跳过当前循环，继续下一个位置

                # 2. 检查是否所有瓷砖1都对应地板0
                # 判断是否存在至少一个瓷砖1对应地板1
                has_valid_overlap = np.any((self.tile == 1) & (floor_region == 1))
                if not has_valid_overlap:
                    print("所有瓷砖1都对应地板0，撤回瓷砖。")
                    self.logs.append(f"无法在位置 ({i}, {j}) 放置瓷砖：所有瓷砖1都对应地板0")
                    continue  # 跳过当前循环，继续下一个位置

                # 3. 确定需要加1的位置：瓷砖1且地板为1
                increment_positions = (self.tile == 1) & (floor_region >= 1)
                print("需要加1的位置:")
                print(increment_positions)

                # 4. 更新地板区域：在需要加1的位置进行累加
                if np.any(increment_positions):
                    # 使用累加而不是替换
                    floor_with_tiles[i:i + tile_rows, j:j + tile_cols][increment_positions] += 1
                    print("瓷砖已成功贴上。更新后的地板区域:")
                    print(floor_with_tiles[i:i + tile_rows, j:j + tile_cols])

                    # 记录铺设位置和日志
                    self.tile_positions.append((i, j))
                    self.logs.append(f"在位置 ({i}, {j}) 放置了瓷砖")
                else:
                    print("没有有效位置可以加1，无法铺设瓷砖。")
                    self.logs.append(f"无法在位置 ({i}, {j}) 放置瓷砖：没有有效位置可以加1")

        # 更新对象的地板矩阵和剩余瓷砖位置
        self.tiled_floor = floor_with_tiles
        self.tiles_remaining = set(self.tile_positions)

        print("铺设完成。")
        print("更新后的地板矩阵：")
        print(self.tiled_floor)
        print("铺设日志：")
        for log in self.logs:
            print(log)
        print("已铺设瓷砖的位置：", self.tile_positions)

    def remove_tiles(self):
        """
        按从中间向两边的顺序尝试移除瓷砖，记录移除日志和状态。
        """
        num_tiles = len(self.tile_positions)
        if num_tiles == 0:
            self.remove_logs.append("没有铺设的瓷砖可以移除。")
            return

        middle_index = num_tiles // 2
        indices = []
        left = middle_index
        right = middle_index + 1

        # 创建一个索引列表，从中间向两边扩展
        while left >= 0 or right < num_tiles:
            if left >= 0:
                indices.append(left)
                left -= 1
            if right < num_tiles:
                indices.append(right)
                right += 1

        tile_rows, tile_cols = self.tile.shape
        for idx in indices:
            if idx >= num_tiles:
                continue
            i, j = self.tile_positions[idx]
            # 暂时移除瓷砖
            self.tiled_floor[i:i + tile_rows, j:j + tile_cols] -= self.tile
            # 检查移除后对应位置的值是否仍然大于等于2
            floor_section = self.tiled_floor[i:i + tile_rows, j:j + tile_cols]
            if np.all(floor_section[self.tile == 1] >= 2):
                self.remove_logs.append(f"位置 ({i}, {j})：瓷砖被移除。")
                # 移除成功，记录当前状态
                self.frames.append(self.tiled_floor.copy())
                # 从剩余瓷砖位置集合中移除该瓷砖
                self.tiles_remaining.remove((i, j))
            else:
                # 无法移除，恢复原状
                self.tiled_floor[i:i + tile_rows, j:j + tile_cols] += self.tile
                self.remove_logs.append(f"位置 ({i}, {j})：无法移除瓷砖，下面的区域会暴露。")

    def calculate_tsp_path(self):
        """
        计算已铺设瓷砖的旅行商路径。

        :return: 路径坐标数组。
        """
        tile_centers = []
        tile_rows, tile_cols = self.tile.shape
        for i, j in self.tiles_remaining:
            center_i = i + tile_rows / 2
            center_j = j + tile_cols / 2
            tile_centers.append((center_j, center_i))  # (x, y)

        if not tile_centers:
            return np.array([])

        tile_centers = np.array(tile_centers)
        dist_matrix = distance_matrix(tile_centers, tile_centers)
        G = nx.complete_graph(len(tile_centers))

        for u, v in G.edges():
            G[u][v]['weight'] = dist_matrix[u][v]

        path = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)
        path_coords = tile_centers[path]

        return path_coords

    def visualize(self, path_coords: np.ndarray):
        """
        可视化铺设情况和旅行商路径。

        :param path_coords: 路径坐标数组。
        """
        if path_coords.size == 0:
            print("没有瓷砖铺设，无法进行可视化。")
            return

        # 第一张图：地板瓷砖布局和最短移动路径
        plt.figure(figsize=(8, 8))
        plt.imshow(self.floor_plan, cmap='gray', origin='lower')

        # 标注瓷砖位置和坐标
        tile_rows, tile_cols = self.tile.shape
        for idx, (x, y) in enumerate(path_coords):
            plt.scatter(x, y, c='red')
            plt.text(x, y, f'({int(y - tile_rows / 2)}, {int(x - tile_cols / 2)})',
                     color='blue', fontsize=8)

        # 绘制移动路径
        plt.plot(path_coords[:, 0], path_coords[:, 1], c='green',
                 linestyle='--', marker='o')

        plt.title('地板瓷砖布局和最短移动路径')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.gca().invert_yaxis()  # 翻转 y 轴以匹配矩阵坐标
        plt.show()

        # 第二张图：覆盖情况和路径规划
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=self.tiled_floor.min(), vmax=self.tiled_floor.max())
        cax = ax.imshow(self.tiled_floor, cmap=cmap, norm=norm, origin='lower')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # 在每个格子上标注数值
        for (i, j), val in np.ndenumerate(self.tiled_floor):
            if self.floor_plan[i, j] > 0:
                ax.text(j, i, int(val), ha='center', va='center',
                        color='white' if val > self.tiled_floor.max() / 2 else 'black',
                        fontsize=8)

        # 标注瓷砖中心位置
        for idx, (x, y) in enumerate(path_coords):
            ax.scatter(x, y, c='red')
            ax.text(x, y, f'({int(y - tile_rows / 2)}, {int(x - tile_cols / 2)})',
                    color='yellow', fontsize=8)

        # 绘制移动路径
        ax.plot(path_coords[:, 0], path_coords[:, 1], c='green',
                linestyle='--', marker='o')

        ax.set_title('针卡覆盖情况和最短移动路径')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.invert_yaxis()  # 翻转 y 轴以匹配矩阵坐标
        plt.show()

    def run(self):
        """
        运行铺设、移除、计算路径和可视化的完整流程。
        """
        # 铺设瓷砖
        self.place_tiles()
        # for log in self.logs:
        #     print(log)
        print("\n铺设完成后的地板矩阵：")
        print(self.tiled_floor)

        # 移除瓷砖
        self.remove_tiles()
        # for log in self.remove_logs:
        #     print(log)
        print("\n最终放置瓷砖的位置：")
        for position in self.tiles_remaining:
            print(position)
        print(f"瓷砖总数：{len(self.tiles_remaining)}")

        # 计算旅行商路径
        path_coords = self.calculate_tsp_path()

        # 可视化
        self.visualize(path_coords)

        # 输出瓷砖总数
        print(f"瓷砖总数：{len(self.tiles_remaining)}")


# 示例使用
if __name__ == "__main__":
    plt.rc("font", family="Microsoft YaHei")

    # 定义地板矩阵（示例：30x30 的圆形区域）
    # 定义参数
    outer_size = 29  # 外圈大小
    overall_size = 28  # 整体矩阵大小
    inner_size = 24  # 内圈大小

    # 创建 MatrixRing 实例
    ring = MatrixRing(outer_size, overall_size, inner_size)

    # 创建矩阵
    matrix = ring.create_matrix()
    print("生成的矩阵:")
    print(matrix)

    # 可视化矩阵
    ring.visualize()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 定义网格参数
    width = 30  # 列数
    height = 30  # 行数
    cell_size = 25 # 每个格子的大小（像素）

    # 创建 GridSelector 实例
    grid_selector = GridSelector(width, height, cell_size)

    # 显示网格界面并获取用户选择的矩阵
    final_matrix = grid_selector.show()

    # 输出生成的矩阵
    print("最终生成的矩阵:")
    print(final_matrix)

    # 截取包含所有1的最小矩阵
    trimmed_matrix = grid_selector.trim_matrix()

    # 输出截取后的矩阵
    print("截取后的最小矩阵:")
    print(trimmed_matrix)

    matrix=pad_matrix_with_zeros(matrix,trimmed_matrix)
    # 创建 FloorPlanner 实例并运行
    planner = FloorPlanner(matrix, trimmed_matrix)
    planner.run()
