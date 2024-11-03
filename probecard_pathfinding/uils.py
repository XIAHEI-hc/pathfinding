import numpy as np


def pad_matrix_with_zeros(base_matrix, padding_matrix):
    """
    在基矩阵周围扩展零填充区域，保持基矩阵不变。

    参数:
    base_matrix (numpy.ndarray): 原始基矩阵，尺寸为80x80。
    padding_matrix (numpy.ndarray): 用于确定扩展大小的矩阵，尺寸为10x5。

    返回:
    numpy.ndarray: 新的扩展后矩阵，边缘填充为0。
    """
    # 获取基矩阵和扩展矩阵的尺寸
    base_rows, base_cols = base_matrix.shape
    pad_rows, pad_cols = padding_matrix.shape

    # 计算新矩阵的尺寸
    new_rows = base_rows + 2 * pad_rows
    new_cols = base_cols + 2 * pad_cols

    # 初始化新矩阵为全零
    new_matrix = np.zeros((new_rows, new_cols), dtype=base_matrix.dtype)

    # 将基矩阵放置在新矩阵的中心
    new_matrix[pad_rows:pad_rows + base_rows, pad_cols:pad_cols + base_cols] = base_matrix

    return new_matrix


# 示例使用
if __name__ == "__main__":
    # 创建80x80的基矩阵（例如全1）
    base = np.ones((20, 20), dtype=int)

    # 创建10x5的扩展矩阵（内容无关紧要，这里仅用于确定扩展大小）
    padding = np.zeros((2, 5), dtype=int)  # 实际上内容不使用

    # 扩展矩阵
    expanded = pad_matrix_with_zeros(base, padding)

    print("新矩阵的尺寸:", expanded.shape)
    print(expanded)

    # 可视化验证（可选）
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(expanded, cmap='gray', interpolation='none')
    plt.title('扩展后的矩阵')
    plt.colorbar()
    plt.show()
