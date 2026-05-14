import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义时期和路径
periods = ['初唐', '盛唐', '中唐', '晚唐']
base_path = '按时期整理网络/整理的网络/'  # 请根据实际路径调整

# 1. 读取所有时期的数据并汇总指标
print("=" * 60)
print("唐诗意象网络四时期对比分析")
print("=" * 60)

summary_data = []

for period in periods:
    print(f"\n--- 正在处理: {period} ---")

    # 读取网络指标（从node_centralities.csv可以获取节点信息）
    node_path = f"{base_path}{period}/node_centralities.csv"
    comm_path = f"{base_path}{period}/community_assignment.csv"
    edge_path = f"{base_path}{period}/cleaned_poetry_pairs.csv"

    # 读取数据
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edge_path)
    comm_df = pd.read_csv(comm_path)

    # 构建网络
    G = nx.Graph()

    # 添加节点（带社区属性）
    comm_dict = dict(zip(comm_df['phrase'], comm_df['community']))
    for node in nodes_df['phrase']:
        G.add_node(node, community=comm_dict.get(node, -1))

    # 添加边
    for _, row in edges_df.iterrows():
        G.add_edge(row['词1'], row['词2'], weight=row['NPMI值'])

    # 计算网络指标
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    degree_std = np.std(degrees)

    # 聚类系数（取最大连通分量，避免孤立节点影响）
    if nx.is_connected(G):
        avg_clustering = nx.average_clustering(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        avg_clustering = nx.average_clustering(G_largest)

    # 模块度（从community_all_sorted_words.csv可以获取社区信息）
    sorted_words_path = f"{base_path}{period}/community_all_sorted_words.csv"
    sorted_df = pd.read_csv(sorted_words_path)
    n_communities = sorted_df['社区ID'].nunique()

    # 从输出文件提取模块度（之前运行时应该有打印，这里暂时用社区数近似）
    # 更准确的做法是从之前的运行日志获取，这里用社区数作为参考

    # 保存指标
    summary_data.append({
        '时期': period,
        '诗歌数量': len(pd.read_csv(f"{base_path}{period}/poems_count.csv")) if os.path.exists(
            f"{base_path}{period}/poems_count.csv") else '未知',
        '节点数': n_nodes,
        '边数': n_edges,
        '网络密度': round(density, 4),
        '平均度': round(avg_degree, 2),
        '最大度': max_degree,
        '度标准差': round(degree_std, 2),
        '平均聚类系数': round(avg_clustering, 3),
        '社区数': n_communities
    })

# 2. 生成对比表格
summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 60)
print("四时期网络指标对比")
print("=" * 60)
print(summary_df.to_string(index=False))

# 保存表格
summary_df.to_csv('four_periods_comparison.csv', index=False, encoding='utf-8-sig')
print("\n✓ 对比表格已保存至: four_periods_comparison.csv")

# 3. 核心意象排名变化分析
print("\n" + "=" * 60)
print("核心意象度排名演变")
print("=" * 60)

# 提取每个时期度排名前20的意象
rank_data = []
for period in periods:
    node_path = f"{base_path}{period}/node_centralities.csv"
    nodes_df = pd.read_csv(node_path)

    # 按度排序，取前20
    top20 = nodes_df.nlargest(20, 'degree')[['phrase', 'degree']]
    top20['时期'] = period
    top20['排名'] = range(1, 21)
    rank_data.append(top20)

rank_df = pd.concat(rank_data, ignore_index=True)

# 创建透视表，看每个意象在各个时期的排名
pivot_rank = rank_df.pivot_table(index='phrase', columns='时期', values='排名', aggfunc='first')

# 只保留至少在两个时期出现过的意象
pivot_rank = pivot_rank.dropna(thresh=2)

# 按初唐排名排序
if '初唐' in pivot_rank.columns:
    pivot_rank = pivot_rank.sort_values('初唐')

print("\n核心意象排名演变（数值为排名，NaN表示未进入前20）:")
print(pivot_rank.to_string())

# 保存排名演变表
pivot_rank.to_csv('core_imageries_rank_evolution.csv', encoding='utf-8-sig')
print("\n✓ 排名演变表已保存至: core_imageries_rank_evolution.csv")

# 4. 关键意象时序追踪（以杨柳为例）
print("\n" + "=" * 60)
print("关键意象'杨柳'的时序追踪")
print("=" * 60)

willow_data = []
for period in periods:
    node_path = f"{base_path}{period}/node_centralities.csv"
    edge_path = f"{base_path}{period}/cleaned_poetry_pairs.csv"
    comm_path = f"{base_path}{period}/community_assignment.csv"

    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edge_path)
    comm_df = pd.read_csv(comm_path)

    # 检查杨柳是否存在
    willow_node = nodes_df[nodes_df['phrase'] == '杨柳']
    if len(willow_node) == 0:
        print(f"{period}: 未找到'杨柳'")
        continue

    # 获取杨柳的度
    degree = willow_node.iloc[0]['degree']

    # 获取杨柳的社区
    willow_comm = comm_df[comm_df['phrase'] == '杨柳']['community'].values[0]

    # 计算杨柳的邻居及其社区分布
    willow_edges = edges_df[(edges_df['词1'] == '杨柳') | (edges_df['词2'] == '杨柳')]

    # 提取邻居列表
    neighbors = []
    for _, row in willow_edges.iterrows():
        if row['词1'] == '杨柳':
            neighbors.append(row['词2'])
        else:
            neighbors.append(row['词1'])

    n_neighbors = len(neighbors)

    # 获取邻居的社区分布
    neighbor_comms = []
    for neighbor in neighbors:
        comm = comm_df[comm_df['phrase'] == neighbor]['community'].values
        if len(comm) > 0:
            neighbor_comms.append(comm[0])

    # 计算跨社区连接数
    unique_comms = set(neighbor_comms)
    cross_community = len(unique_comms)

    # 计算平均关联强度
    avg_weight = willow_edges['NPMI值'].mean()

    # 获取最强关联（前3）
    top3 = willow_edges.nlargest(3, 'NPMI值')
    top3_list = []
    for _, row in top3.iterrows():
        if row['词1'] == '杨柳':
            top3_list.append(f"{row['词2']}({row['NPMI值']:.3f})")
        else:
            top3_list.append(f"{row['词1']}({row['NPMI值']:.3f})")

    willow_data.append({
        '时期': period,
        '度': degree,
        '邻居数': n_neighbors,
        '跨社区连接数': cross_community,
        '平均关联强度': round(avg_weight, 3),
        '最强关联': '; '.join(top3_list),
        '所属社区': willow_comm
    })

willow_df = pd.DataFrame(willow_data)
print("\n'杨柳'四个时期演化:")
print(willow_df.to_string(index=False))

# 保存杨柳追踪数据
willow_df.to_csv('willow_temporal_tracking.csv', index=False, encoding='utf-8-sig')
print("\n✓ 杨柳追踪数据已保存至: willow_temporal_tracking.csv")

# 5. 生成可视化图表

# 5.1 网络规模演化图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 边数演化
axes[0, 0].plot(summary_df['时期'], summary_df['边数'], 'o-', linewidth=2, markersize=8, color='steelblue')
axes[0, 0].set_xlabel('时期')
axes[0, 0].set_ylabel('边数')
axes[0, 0].set_title('网络边数演化')
axes[0, 0].grid(True, alpha=0.3)

# 平均度演化
axes[0, 1].plot(summary_df['时期'], summary_df['平均度'], 'o-', linewidth=2, markersize=8, color='coral')
axes[0, 1].set_xlabel('时期')
axes[0, 1].set_ylabel('平均度')
axes[0, 1].set_title('平均度演化')
axes[0, 1].grid(True, alpha=0.3)

# 网络密度演化
axes[0, 2].plot(summary_df['时期'], summary_df['网络密度'], 'o-', linewidth=2, markersize=8, color='mediumseagreen')
axes[0, 2].set_xlabel('时期')
axes[0, 2].set_ylabel('网络密度')
axes[0, 2].set_title('网络密度演化')
axes[0, 2].grid(True, alpha=0.3)

# 聚类系数演化
axes[1, 0].plot(summary_df['时期'], summary_df['平均聚类系数'], 'o-', linewidth=2, markersize=8, color='purple')
axes[1, 0].set_xlabel('时期')
axes[1, 0].set_ylabel('平均聚类系数')
axes[1, 0].set_title('聚类系数演化')
axes[1, 0].grid(True, alpha=0.3)

# 最大度演化
axes[1, 1].plot(summary_df['时期'], summary_df['最大度'], 'o-', linewidth=2, markersize=8, color='goldenrod')
axes[1, 1].set_xlabel('时期')
axes[1, 1].set_ylabel('最大度')
axes[1, 1].set_title('最大度演化')
axes[1, 1].grid(True, alpha=0.3)

# 社区数演化
axes[1, 2].plot(summary_df['时期'], summary_df['社区数'], 'o-', linewidth=2, markersize=8, color='tomato')
axes[1, 2].set_xlabel('时期')
axes[1, 2].set_ylabel('社区数')
axes[1, 2].set_title('社区数量演化')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('唐诗意象网络四时期演化趋势', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('network_evolution_trends.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 演化趋势图已保存至: network_evolution_trends.png")

# 5.2 核心意象排名热力图
plt.figure(figsize=(12, 10))

# 选取在多个时期都进入前20的意象
core_phrases = []
for period in periods:
    node_path = f"{base_path}{period}/node_centralities.csv"
    nodes_df = pd.read_csv(node_path)
    core_phrases.extend(nodes_df.nlargest(20, 'degree')['phrase'].tolist())

# 统计出现次数，选取至少出现在两个时期的意象
from collections import Counter

phrase_counts = Counter(core_phrases)
stable_phrases = [p for p, c in phrase_counts.items() if c >= 2]

# 构建排名矩阵
rank_matrix = []
for phrase in stable_phrases[:15]:  # 取前15个
    row = []
    for period in periods:
        node_path = f"{base_path}{period}/node_centralities.csv"
        nodes_df = pd.read_csv(node_path)
        top20 = nodes_df.nlargest(20, 'degree')['phrase'].tolist()
        if phrase in top20:
            rank = top20.index(phrase) + 1
            row.append(rank)
        else:
            row.append(np.nan)
    rank_matrix.append(row)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(rank_matrix, annot=True, fmt='.0f', cmap='YlOrRd_r',
            xticklabels=periods, yticklabels=stable_phrases[:15],
            cbar_kws={'label': '排名（越小越重要）'})
plt.title('核心意象排名演变热力图', fontsize=14, fontweight='bold')
plt.xlabel('时期')
plt.ylabel('意象')
plt.tight_layout()
plt.savefig('core_imageries_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 核心意象热力图已保存至: core_imageries_heatmap.png")

# 5.3 杨柳演化折线图
if len(willow_df) > 0:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 度演化
    color = 'tab:red'
    ax1.set_xlabel('时期')
    ax1.set_ylabel('度', color=color)
    ax1.plot(willow_df['时期'], willow_df['度'], 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)

    # 平均关联强度演化（双y轴）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('平均关联强度', color=color)
    ax2.plot(willow_df['时期'], willow_df['平均关联强度'], 's--', color=color, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('“杨柳”意象的时序演化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('willow_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ 杨柳演化图已保存至: willow_evolution.png")

print("\n" + "=" * 60)
print("✓ 所有分析完成！共生成以下文件：")
print("  - four_periods_comparison.csv（四时期指标对比）")
print("  - core_imageries_rank_evolution.csv（核心意象排名演变）")
print("  - willow_temporal_tracking.csv（杨柳时序追踪）")
print("  - network_evolution_trends.png（网络指标演化图）")
print("  - core_imageries_heatmap.png（核心意象热力图）")
print("  - willow_evolution.png（杨柳演化图）")
print("=" * 60)
