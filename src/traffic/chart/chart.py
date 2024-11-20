import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def draw_hist(df, output_dir):
    # 设置A4纸张大小（单位：英寸）
    a4_width_inches = 8.27
    a4_height_inches = 11.69

    # 设置每页的列数和行数
    n_cols = 3
    n_rows_per_page = 7  # 根据需要调整，确保每页能容纳足够的子图

    # 计算总页数
    total_columns = len(df.columns)
    pages = (total_columns - 1) // (n_cols * n_rows_per_page) + 1

    for page in range(pages):
        # 创建新的图形对象
        fig, axes = plt.subplots(nrows=n_rows_per_page, ncols=n_cols, figsize=(a4_width_inches, a4_height_inches))
        # 逐列绘制直方图
        for i in range(n_cols * n_rows_per_page):
            col_index = page * n_cols * n_rows_per_page + i
            if col_index < total_columns:
                ax = axes[i // n_cols, i % n_cols]
                col = df.columns[col_index]
                ax.hist(df[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax.set_title(col, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # 如果没有更多的列，则删除多余的子图
                fig.delaxes(axes[i // n_cols, i % n_cols])
        # 调整子图之间的间距
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        # 保存当前页面的图像
        plt.savefig(f'{output_dir}/hist_page_{page + 1}.png', dpi=300)
        plt.close(fig)  # 关闭当前图形，释放内存
    logging.info(f"Saved {pages}pages of hist charts in {output_dir}.")


def draw_line(df, output_dir):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 将时间戳精度设置为1分钟
    df['timestamp'] = df['timestamp'].dt.floor('1min')
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'item_id']  # 排除item_id列
    # 设置A4纸大小（单位为英寸）
    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    ROWS, COLS = 4, 3
    # 计算需要的页数
    charts_per_page = ROWS * COLS
    total_pages = math.ceil(len(numeric_columns) / charts_per_page)

    # 为每个item_id分配一个颜色
    unique_item_ids = df['item_id'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_item_ids)))
    color_dict = dict(zip(unique_item_ids, colors))

    # 遍历每一页
    for page in range(total_pages):
        # 创建一个新的图形，大小为A4，并为图例留出空间
        fig, axes = plt.subplots(ROWS, COLS, figsize=(A4_WIDTH, A4_HEIGHT))
        fig.suptitle(f'Page {page + 1}', fontsize=12)

        # 获取当前页面要绘制的列
        start_idx = page * charts_per_page
        end_idx = min((page + 1) * charts_per_page, len(numeric_columns))
        page_columns = numeric_columns[start_idx:end_idx]

        # 遍历当前页面的每个数值列
        for idx, column in enumerate(page_columns):
            row = idx // COLS
            col = idx % COLS
            ax = axes[row, col]

            # 按item_id分组并绘图
            for item_id, group in df.groupby('item_id'):
                color = color_dict[item_id]

                # 对每个时间戳取平均值并进行2分钟重采样
                resampled = group.set_index('timestamp').resample('2T')[column].mean().dropna()

                if len(resampled) > 11:  # 确保有足够的点来进行滤波
                    # 使用Savitzky-Golay滤波器去除异常点
                    smooth_values = savgol_filter(resampled.values, window_length=11, polyorder=3)

                    # 创建时间序列索引
                    x = np.arange(len(resampled))

                    # 多项式拟合
                    z = np.polyfit(x, smooth_values, 3)
                    p = np.poly1d(z)

                    # 绘制原始数据点（更不明显）
                    ax.scatter(resampled.index, resampled.values, s=5, alpha=0.2, color=color)

                    # 绘制平滑曲线（更突出）
                    ax.plot(resampled.index, smooth_values, color=color, alpha=1, linewidth=2)

                    # 绘制拟合曲线
                    ax.plot(resampled.index, p(x), color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                else:
                    # 如果点数不足，只绘制原始数据点
                    ax.scatter(resampled.index, resampled.values, s=5, alpha=0.2, color=color)

            ax.set_title(f'{column} Over Time')

            # 只显示每页最左侧子图的y轴和最下方子图的x轴
            if col != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            if row != ROWS - 1:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.tick_params(axis='x', rotation=45)

        # 删除多余的子图
        for idx in range(len(page_columns), ROWS * COLS):
            row = idx // COLS
            col = idx % COLS
            fig.delaxes(axes[row, col])

        # 添加图例
        handles = [plt.Line2D([0], [0], color=color, lw=2, label=f'ID: {item_id}') for item_id, color in
                   color_dict.items()]
        fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)

        # 保存为PNG文件
        plt.savefig(f'{output_dir}/line_page_{page + 1}.pdf',  bbox_inches='tight')
        plt.close(fig)
    logging.info(f"Saved {total_pages}pages of line charts in {output_dir}.")