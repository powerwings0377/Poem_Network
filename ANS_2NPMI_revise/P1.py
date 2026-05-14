#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算原始网络模块度 Q_original 与随机图模块度 Q_random 的对比。
输入：边列表 CSV (词1, 词2, ...)
输出：控制台打印均值、标准差，并可选保存直方图。
"""

import pandas as pd
import networkx as nx
import numpy as np
from community import community_louvain  # python-louvain
import matplotlib.pyplot as plt
import time

# ======================= 配置参数 =======================
EDGES_CSV = "重新生成Gephi图/cleaned_poetry_pairs.csv"  # 你的边文件
RANDOM_COUNT = 100  # 生成随机图的数量
SAVE_HISTOGRAM = True  # 是否保存直方图
HISTOGRAM_FILE = "q_random_histogram.png"


# ========================================================

def load_graph_from_csv(csv_path):
    """从 CSV 中读取 '词1', '词2' 列，构建无向无权图"""
    print(f"正在读取边文件: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {'词1', '词2'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 文件缺少列 {required - set(df.columns)}")
    G = nx.Graph()
    # 逐行添加边（忽略权重）
    for _, row in df.iterrows():
        u = str(row['词1']).strip()
        v = str(row['词2']).strip()
        G.add_edge(u, v)
    print(f"图已构建: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    return G


def compute_modularity(G):
    """返回 Louvain 算法得到的社区划分及模块度 Q"""
    partition = community_louvain.best_partition(G)
    q = community_louvain.modularity(partition, G)
    return q, partition


def generate_random_graphs(G, n=100):
    """
    生成 n 个随机图，每个图保持与 G 相同的度数序列（使用 double edge swap）。
    使用 networkx 的 random_reference 方法（交换边）保证简单图。
    """
    print(f"\n正在生成 {n} 个随机图（保持度数序列）...")
    random_graphs = []
    start = time.time()
    for i in range(n):
        # random_reference 通过随机交换边来打乱图，保持每个节点的度数
        G_rand = nx.random_reference(G, connectivity=False)
        random_graphs.append(G_rand)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  已生成 {i + 1}/{n} 个，耗时 {elapsed:.1f} 秒")
    print(f"生成完成，共耗时 {time.time() - start:.1f} 秒")
    return random_graphs


def compute_random_modularities(random_graphs):
    """计算一批随机图的模块度"""
    q_list = []
    for i, G_rand in enumerate(random_graphs):
        # 对随机图运行 Louvain 并计算模块度
        part = community_louvain.best_partition(G_rand)
        q = community_louvain.modularity(part, G_rand)
        q_list.append(q)
        if (i + 1) % 20 == 0:
            print(f"  已计算 {i + 1}/{len(random_graphs)} 个随机图的模块度")
    return np.array(q_list)


def main():
    # 1. 加载原始图
    G = load_graph_from_csv(EDGES_CSV)

    # 2. 原始模块度
    print("\n计算原始网络模块度...")
    Q_orig, partition_orig = compute_modularity(G)
    print(f"Q_original = {Q_orig:.6f}")

    # 3. 生成随机图
    rand_graphs = generate_random_graphs(G, n=RANDOM_COUNT)

    # 4. 计算随机图模块度
    print("\n计算随机图模块度...")
    Q_rand_list = compute_random_modularities(rand_graphs)
    Q_rand_mean = np.mean(Q_rand_list)
    Q_rand_std = np.std(Q_rand_list)

    # 5. 输出结果
    print("\n========== 最终结果 ==========")
    print(f"原始模块度 Q_original     : {Q_orig:.6f}")
    print(f"随机图模块度 (均值 ± 标准差): {Q_rand_mean:.6f} ± {Q_rand_std:.6f}")
    print(f"差异 (原始 - 随机)        : {Q_orig - Q_rand_mean:.6f}")

    # 6. 可选：绘制直方图
    if SAVE_HISTOGRAM:
        plt.figure(figsize=(8, 5))
        plt.hist(Q_rand_list, bins=30, alpha=0.7, edgecolor='black', label='Random graphs')
        plt.axvline(Q_orig, color='red', linestyle='--', linewidth=2, label=f'Q_original = {Q_orig:.4f}')
        plt.xlabel("Modularity Q")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Q for {RANDOM_COUNT} random graphs (degree-preserving)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(HISTOGRAM_FILE, dpi=300, bbox_inches='tight')
        print(f"\n直方图已保存为: {HISTOGRAM_FILE}")
        # plt.show()  # 如果希望显示，取消注释

    # 7. 简单判断
    if Q_orig > Q_rand_mean + 3 * Q_rand_std:
        print("\n结论: 原始模块度显著高于随机图的模块度（>3σ），社区结构非随机。")
    elif Q_orig > Q_rand_mean + 2 * Q_rand_std:
        print("\n结论: 原始模块度明显高于随机图（>2σ），支持存在非随机社区结构。")
    else:
        print("\n结论: 原始模块度与随机图差异不显著，需谨慎解释。")


if __name__ == "__main__":
    main()