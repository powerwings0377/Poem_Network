
import csv
import re

# 1. 读取 CSV 文件，建立节点 -> degree 的映射
node_degree = {}
with open('new_100_output/full_network_degree_ranking.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
        node = row[0].strip()
        degree = int(row[1])
        node_degree[node] = degree

# 2. 解析 communities_list.txt，提取每个社区的节点列表
communities = []
current_community = None

with open('new_100/communities_list.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('社区'):
            # 新社区开始，格式如 "社区 0 (194 个节点):"
            match = re.search(r'社区 (\d+)', line)
            if match:
                if current_community is not None:
                    communities.append(current_community)
                comm_id = int(match.group(1))
                current_community = {'id': comm_id, 'nodes': []}
        elif current_community is not None and line and not line.startswith('===') and not line.startswith('共发现'):
            # 解析节点列表，节点之间以逗号+空格分隔
            # 例如 "  东西, 南北, 今年, 登高, ..."
            # 去掉开头的空格，按 ", " 分割
            parts = line.split(', ')
            for part in parts:
                node = part.strip()
                if node:  # 非空
                    current_community['nodes'].append(node)
    # 添加最后一个社区
    if current_community is not None:
        communities.append(current_community)

# 3. 为每个社区内的节点按 degree 排序并输出到 txt 文件
with open('new_100_output/community_degree_ranking.txt', 'w', encoding='utf-8') as out:
    for comm in communities:
        comm_id = comm['id']
        nodes = comm['nodes']
        # 获取每个节点的 degree，跳过未在 CSV 中出现的节点（理论上都应该存在）
        node_deg_list = [(node, node_degree.get(node, 0)) for node in nodes]
        # 按 degree 降序排序
        node_deg_list.sort(key=lambda x: x[1], reverse=True)
        out.write(f"社区 {comm_id} (共 {len(nodes)} 个节点)\n")
        out.write("-" * 50 + "\n")
        for node, deg in node_deg_list:
            out.write(f"{node}: {deg}\n")
        out.write("\n")

print("已生成 community_degree_ranking.txt")