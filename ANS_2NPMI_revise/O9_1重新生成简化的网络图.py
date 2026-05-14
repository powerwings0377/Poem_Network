"""
导出简化网络供 Gephi 使用
新增：导出每个词的度排名表格
输出文件：
1. figure5_simplified_for_gephi.gexf
2. figure5_simplified_for_gephi.graphml
3. word_degree_ranking.csv （词的度排名）
"""

import pandas as pd
import networkx as nx

# ================== 配置 ==================
EDGES_FILE = "重新生成Gephi图/cleaned_poetry_pairs.csv"
COMMUNITY_FILE = "重新生成Gephi图/community_assignment.csv"
OUTPUT_GEXF = "重新生成Gephi图/figure5_simplified_for_gephi.gexf"
OUTPUT_GRAPHML = "重新生成Gephi图/figure5_simplified_for_gephi.graphml"
OUTPUT_RANKING = "重新生成Gephi图/word_degree_ranking.csv"  # 度排名输出文件

# 筛选参数（与 generate_figure2.py 中的 simplified 网络一致）
DEGREE_THRESHOLD = 50        # 全局度阈值
TOP_K_IN_COMMUNITY = 30      # 每个社区内保留的节点数（按社区内度排序）

# ================== 读取数据 ==================
print("读取边数据...")
edges_df = pd.read_csv(EDGES_FILE)
print(f"  边数: {len(edges_df)}")

print("读取社区分配...")
comm_df = pd.read_csv(COMMUNITY_FILE)
print(f"  节点数: {len(comm_df)}")

# 构建完整网络
G_full = nx.Graph()

# 添加节点和社区属性
phrase_to_comm = {}
for _, row in comm_df.iterrows():
    phrase = str(row['phrase']).strip()
    comm = int(row['community'])
    phrase_to_comm[phrase] = comm
    G_full.add_node(phrase, community=comm)

# 添加边（保留权重）
for _, row in edges_df.iterrows():
    u = str(row['词1']).strip()
    v = str(row['词2']).strip()
    w = row['NPMI值']
    if u in phrase_to_comm and v in phrase_to_comm:
        G_full.add_edge(u, v, weight=w)

print(f"完整网络: {G_full.number_of_nodes()} 节点, {G_full.number_of_edges()} 边")

# ================== 筛选重要节点 ==================
# 全局度
degree_all = dict(G_full.degree())

# 社区内度（仅统计同一社区内部的边）
communities = {}
for node, comm in phrase_to_comm.items():
    communities.setdefault(comm, []).append(node)

# 记录每个节点的社区内度
inner_degree = {}
for comm, nodes in communities.items():
    subG = G_full.subgraph(nodes)
    deg_in = dict(subG.degree())
    for n, d in deg_in.items():
        inner_degree[n] = d

# 为每个社区选择重要的节点
important_nodes = set()
for comm, nodes in communities.items():
    # 按社区内度降序排序
    sorted_nodes = sorted(nodes, key=lambda x: inner_degree.get(x, 0), reverse=True)
    # 取前 TOP_K_IN_COMMUNITY 个，且全局度 > DEGREE_THRESHOLD
    for node in sorted_nodes[:TOP_K_IN_COMMUNITY]:
        if degree_all.get(node, 0) > DEGREE_THRESHOLD:
            important_nodes.add(node)

print(f"筛选后重要节点数: {len(important_nodes)}")

# 提取子图
G_simple = G_full.subgraph(important_nodes).copy()
print(f"简化网络: {G_simple.number_of_nodes()} 节点, {G_simple.number_of_edges()} 边")

# ================== 添加额外的节点属性 ==================
for node in G_simple.nodes():
    G_simple.nodes[node]['degree_all'] = degree_all[node]
    G_simple.nodes[node]['inner_degree'] = inner_degree.get(node, 0)
    G_simple.nodes[node]['community'] = phrase_to_comm[node]

# 可选：为边添加归一化权重
weights = [data['weight'] for _, _, data in G_simple.edges(data=True)]
if weights:
    min_w, max_w = min(weights), max(weights)
    for u, v, data in G_simple.edges(data=True):
        if max_w > min_w:
            norm = (data['weight'] - min_w) / (max_w - min_w)
        else:
            norm = 0.5
        data['weight_norm'] = norm

# ================== 【新增】导出 每个词的度排名 ==================
print(f"\n正在生成【词语度排名】表格: {OUTPUT_RANKING}")

# 1. 收集所有筛选后节点的信息
ranking_list = []
for node in G_simple.nodes():
    ranking_list.append({
        "词语": node,
        "全局度": degree_all[node],
        "社区内度": inner_degree[node],
        "所属社区": phrase_to_comm[node]
    })

# 2. 按 全局度 降序排序
ranking_df = pd.DataFrame(ranking_list)
ranking_df = ranking_df.sort_values(by="全局度", ascending=False)
ranking_df.reset_index(drop=True, inplace=True)
ranking_df.index = ranking_df.index + 1  # 排名从 1 开始
ranking_df.index.name = "排名"

# 3. 保存为 CSV
ranking_df.to_csv(OUTPUT_RANKING, encoding="utf-8-sig")
print(f"✅ 度排名表格导出完成：{OUTPUT_RANKING}")

# ================== 导出 Gephi 文件 ==================
print(f"\n导出 {OUTPUT_GEXF} ...")
nx.write_gexf(G_simple, OUTPUT_GEXF, version="1.2draft")
print(f"导出成功！可以在 Gephi 中打开 {OUTPUT_GEXF}")

print(f"导出 {OUTPUT_GRAPHML} ...")
nx.write_graphml(G_simple, OUTPUT_GRAPHML)
print("\n🎉 全部完成！")