# -*- coding: utf-8 -*-
"""
时序网络分析 + 度排名输出
1. 计算整体网络的度排名（需要全集边表）
2. 计算四个时期网络的度排名
3. 将全集最佳社区划分投射到四个时期，计算模块度及社区规模
"""

import os
import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt

# ============================================================
# 用户配置：请根据您的实际路径修改
# ============================================================

# 全集最佳划分文件 (来自 new_100 文件夹)
membership_file = r"new_100/community_membership.csv"

# 【新增】全集网络边表文件（完整语料的 cleaned_poetry_pairs.csv）
full_edges_file = r"按时期整理网络/整理的网络/全集/cleaned_poetry_pairs.csv"   # 请修改为实际路径，如果没有可以设为 None

# 四个时期的网络边表文件 (cleaned_poetry_pairs.csv)
period_data = {
    "Early Tang": r"按时期整理网络/整理的网络/初唐/cleaned_poetry_pairs.csv",
    "High Tang":  r"按时期整理网络/整理的网络/盛唐/cleaned_poetry_pairs.csv",
    "Mid Tang":   r"按时期整理网络/整理的网络/中唐/cleaned_poetry_pairs.csv",
    "Late Tang":  r"按时期整理网络/整理的网络/晚唐/cleaned_poetry_pairs.csv",
}

# 输出文件夹
output_dir = "new_100_output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 辅助函数：计算并保存度排名
# ============================================================
def compute_and_save_degree_ranking(G, output_path, graph_name=""):
    """计算图中所有节点的度，按度降序保存到 CSV，并打印前5名"""
    degrees = dict(G.degree())
    df_degree = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    df_degree = df_degree.sort_values('degree', ascending=False).reset_index(drop=True)
    df_degree.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   {graph_name} 度排名已保存至: {output_path}")
    print(f"   前5名: {df_degree.head(5).to_string(index=False)}")
    return df_degree

# ============================================================
# 1. 整体网络的度排名（如果提供了全集边表）
# ============================================================
print("=" * 60)
if full_edges_file and os.path.exists(full_edges_file):
    print("加载整体网络边表...")
    G_full = nx.Graph()
    edges_full = pd.read_csv(full_edges_file)
    for _, row in edges_full.iterrows():
        G_full.add_edge(row['词1'], row['词2'], weight=row['NPMI值'])
    print(f"整体网络: {G_full.number_of_nodes()} 节点, {G_full.number_of_edges()} 边")
    full_rank_path = os.path.join(output_dir, "full_network_degree_ranking.csv")
    compute_and_save_degree_ranking(G_full, full_rank_path, "整体网络")
else:
    print("未提供整体网络边表或文件不存在，跳过整体度排名计算。")

# ============================================================
# 2. 加载全集最佳社区划分
# ============================================================
print("\n" + "=" * 60)
print("加载全集最佳社区划分")
print("=" * 60)
membership_df = pd.read_csv(membership_file)
standard_partition = dict(zip(membership_df['phrase'], membership_df['community']))
print(f"共加载 {len(standard_partition)} 个节点的社区标签")

# ============================================================
# 3. 对每个时期进行处理
# ============================================================
results = []

for period_name, edge_file in period_data.items():
    print(f"\n--- 处理: {period_name} ---")
    if not os.path.exists(edge_file):
        print(f"错误: 找不到文件 {edge_file}")
        continue

    # 读取边表，构建图
    edges_df = pd.read_csv(edge_file)
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['词1'], row['词2'], weight=row['NPMI值'])
    print(f"  网络规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # ---- 【新增】计算该时期的度排名并保存 ----
    period_rank_path = os.path.join(output_dir, f"{period_name.replace(' ', '_')}_degree_ranking.csv")
    compute_and_save_degree_ranking(G, period_rank_path, period_name)

    # 固定划分投射
    current_partition = {}
    for node in G.nodes():
        if node in standard_partition:
            current_partition[node] = standard_partition[node]
        else:
            current_partition[node] = -1

    # 计算固定划分模块度
    q_fixed = community_louvain.modularity(current_partition, G, weight='weight')
    print(f"  固定划分模块度: {q_fixed:.4f}")

    # 统计各社区节点数
    comm_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for node, comm in current_partition.items():
        if comm in comm_counts:
            comm_counts[comm] += 1
    print(f"  社区节点分布: C0={comm_counts[0]}, C1={comm_counts[1]}, C2={comm_counts[2]}, C3={comm_counts[3]}")

    # 追踪杨柳度
    willow_degree = G.degree('杨柳') if '杨柳' in G.nodes() else None
    print(f"  杨柳的度: {willow_degree}")

    results.append({
        'Period': period_name,
        'Modularity_fixed': round(q_fixed, 4),
        'Community0_nodes': comm_counts[0],
        'Community1_nodes': comm_counts[1],
        'Community2_nodes': comm_counts[2],
        'Community3_nodes': comm_counts[3],
        'Willow_degree': willow_degree if willow_degree is not None else 'N/A'
    })

# ============================================================
# 4. 输出汇总表格
# ============================================================
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("四时期固定划分分析结果")
print("=" * 60)
print(results_df.to_string(index=False))

output_csv = os.path.join(output_dir, "temporal_fixed_partition_analysis.csv")
results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✓ 结果已保存至: {output_csv}")

# ============================================================
# 5. 绘制社区规模演化图与模块度演化图（沿用原代码）
# ============================================================
plt.figure(figsize=(10, 6))
periods = results_df['Period'].tolist()
for comm_id in range(4):
    col = f'Community{comm_id}_nodes'
    plt.plot(periods, results_df[col], marker='o', label=f'Community {comm_id}')
plt.xlabel('Period')
plt.ylabel('Number of Nodes')
plt.title('Evolution of Community Sizes (Fixed Partition)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
comm_plot = os.path.join(output_dir, "community_evolution.png")
plt.savefig(comm_plot, dpi=300)
plt.show()
print(f"✓ 社区规模演化图保存至: {comm_plot}")

plt.figure(figsize=(8, 5))
mod_vals = results_df['Modularity_fixed'].tolist()
plt.plot(periods, mod_vals, marker='s', linestyle='--', color='darkred')
plt.xlabel('Period')
plt.ylabel('Modularity Q (fixed partition)')
plt.title('Modularity of Fixed Partition Across Periods')
plt.grid(True, alpha=0.3)
mod_plot = os.path.join(output_dir, "modularity_evolution.png")
plt.savefig(mod_plot, dpi=300)
plt.show()
print(f"✓ 模块度演化图保存至: {mod_plot}")

print("\n" + "=" * 60)
print(f"所有输出已保存到文件夹: {output_dir}")
print("=" * 60)