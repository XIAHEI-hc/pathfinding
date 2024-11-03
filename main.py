import logging
from DIE_map.DIE_map import MatrixRing
from probercard_setting.probecard_UI import GridSelector
from probecard_pathfinding.probecard_pathfinding_no_out_of_wafer import FloorPlanner




def main():
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
    width = 80  # 列数
    height = 60  # 行数
    cell_size = 10  # 每个格子的大小（像素）

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

    # 创建 FloorPlanner 实例并运行
    planner = FloorPlanner(matrix, trimmed_matrix)
    planner.run()



if __name__ == "__main__":
    main()
