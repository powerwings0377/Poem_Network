# generate_figure4_ego_network_chinese.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_yangliu_ego_network():
    """生成'杨柳'的自我中心网络图"""
    print("正在生成图4：'杨柳'意象的自我中心网络...")

    # 1. 加载数据
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    print(f"已加载数据：{len(edges_df)}条边，{len(community_df)}个节点")

    # 2. 创建网络图
    G = nx.Graph()

    # 创建诗语到社区的映射
    community_dict = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        comm_id = int(row['community'])
        community_dict[phrase] = comm_id
        G.add_node(phrase, community=comm_id)

    # 添加边
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in community_dict and target in community_dict:
            G.add_edge(source, target, weight=weight)
            edge_count += 1

    print(f"网络构建完成：{G.number_of_nodes()}个节点，{G.number_of_edges()}条边")

    # 3. 检查'杨柳'是否在网络中
    central_node = "杨柳"
    if central_node not in G.nodes():
        print(f"警告：'{central_node}'不在网络中！")
        # 查找包含"杨"或"柳"的节点
        similar_nodes = [n for n in G.nodes() if "杨" in n or "柳" in n]
        if similar_nodes:
            print(f"找到相似节点：{similar_nodes}")
            central_node = similar_nodes[0]
            print(f"将使用'{central_node}'作为中心节点")
        else:
            # 查找度最高的节点作为替代
            degrees = dict(G.degree())
            top_node = max(degrees.items(), key=lambda x: x[1])[0]
            print(f"未找到相似节点，将使用度最高的节点'{top_node}'")
            central_node = top_node

    # 4. 获取自我中心网络（一度邻居）
    print(f"\n提取'{central_node}'的自我中心网络...")
    ego = nx.ego_graph(G, central_node, radius=1)
    print(f"自我中心网络：{ego.number_of_nodes()}个节点，{ego.number_of_edges()}条边")

    # 5. 准备可视化参数
    # 社区颜色映射（与图2保持一致）
    community_colors = {
        0: '#E74C3C',  # 红色 - 春景闺情
        1: '#3498DB',  # 蓝色 - 羁旅思乡
        2: '#2ECC71',  # 绿色 - 人生际遇
        3: '#9B59B6'  # 紫色 - 隐逸山水
    }

    # 社区中文标签
    community_labels = {
        0: "春景闺情",
        1: "羁旅思乡",
        2: "人生际遇",
        3: "隐逸山水"
    }

    # 节点颜色和大小
    node_colors = []
    node_sizes = []
    node_border_colors = []
    node_border_widths = []

    for node in ego.nodes():
        if node == central_node:
            # 中心节点：金色，大尺寸，粗边框
            node_colors.append('gold')
            node_sizes.append(2800)
            node_border_colors.append('darkorange')
            node_border_widths.append(3.0)
        else:
            # 邻居节点：按社区着色
            comm_id = G.nodes[node].get('community', 0)
            color = community_colors.get(comm_id, '#95A5A6')
            node_colors.append(color)
            node_border_colors.append('white')
            node_border_widths.append(1.5)

            # 节点大小基于与中心节点的NPMI权重
            if central_node in ego and node in ego[central_node]:
                weight = ego[central_node][node]['weight']
                size = 1000 + weight * 1200  # NPMI越高，节点越大
            else:
                size = 800
            node_sizes.append(size)

    # 6. 创建布局
    print("计算网络布局...")

    # 使用spring布局，但将中心节点固定在中心
    pos = nx.spring_layout(ego, seed=42, k=1.5, iterations=150)

    # 确保中心节点在中心位置
    pos[central_node] = (0, 0)

    # 调整邻居节点位置，使其围绕中心节点
    for node in ego.nodes():
        if node != central_node:
            # 稍微调整位置，使布局更美观
            if node in pos:
                x, y = pos[node]
                # 将节点向外推一点
                distance = np.sqrt(x ** 2 + y ** 2)
                if distance > 0:
                    scale = 1.2  # 向外扩展系数
                    pos[node] = (x * scale, y * scale)

    # 7. 绘制网络图
    print("绘制网络图...")
    plt.figure(figsize=(15, 13))

    # 7.1 绘制边（按NPMI权重分级绘制）
    print("  绘制边...")

    # 收集边权重信息
    strong_edges = []  # NPMI > 0.4
    medium_edges = []  # 0.3 < NPMI ≤ 0.4
    weak_edges = []  # NPMI ≤ 0.3

    for u, v, data in ego.edges(data=True):
        weight = data.get('weight', 0.3)
        if weight > 0.4:
            strong_edges.append((u, v, weight))
        elif weight > 0.3:
            medium_edges.append((u, v, weight))
        else:
            weak_edges.append((u, v, weight))

    # 先绘制弱边（作为背景）
    if weak_edges:
        weak_edgelist = [(u, v) for u, v, w in weak_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=weak_edgelist,
                               width=0.8,
                               alpha=0.3,
                               edge_color='lightgray',
                               style='dashed')

    # 绘制中等强度边
    if medium_edges:
        medium_edgelist = [(u, v) for u, v, w in medium_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=medium_edgelist,
                               width=1.5,
                               alpha=0.5,
                               edge_color='gray')

    # 绘制强边（最显著）
    if strong_edges:
        strong_edgelist = [(u, v) for u, v, w in strong_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=strong_edgelist,
                               width=2.5,
                               alpha=0.7,
                               edge_color='darkblue')

    # 7.2 绘制节点
    print("  绘制节点...")
    nx.draw_networkx_nodes(ego, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors=node_border_colors,
                           linewidths=node_border_widths)

    # 7.3 绘制节点标签
    print("  绘制节点标签...")

    # 为所有节点添加标签，但中心节点标签更大更突出
    labels = {}
    for node in ego.nodes():
        if node == central_node:
            # 中心节点：大字号，加粗，特殊颜色
            labels[node] = node
            nx.draw_networkx_labels(ego, pos,
                                    labels={node: node},
                                    font_size=14,
                                    font_weight='bold',
                                    font_color='darkred')
        else:
            # 邻居节点：正常标签
            labels[node] = node

    # 绘制邻居节点标签（避免与中心节点标签重叠）
    neighbor_labels = {node: node for node in ego.nodes() if node != central_node}
    nx.draw_networkx_labels(ego, pos,
                            labels=neighbor_labels,
                            font_size=10,
                            font_weight='normal')

    # 7.4 绘制边权重标签（只标注强关联）
    print("  绘制边权重标签...")
    edge_labels = {}
    for u, v, data in ego.edges(data=True):
        weight = data.get('weight', 0)
        if weight > 0.35:  # 只标注较强的关联
            edge_labels[(u, v)] = f'{weight:.2f}'

    nx.draw_networkx_edge_labels(ego, pos,
                                 edge_labels=edge_labels,
                                 font_size=9,
                                 font_weight='bold',
                                 font_color='darkgreen',
                                 label_pos=0.5,
                                 bbox=dict(alpha=0))

    # 8. 添加图例
    print("添加图例...")
    from matplotlib.lines import Line2D

    # 创建自定义图例句柄
    legend_elements = [
        Patch(facecolor='gold', edgecolor='darkorange', linewidth=3,
              label=f'中心节点: "{central_node}"'),
        Line2D([0], [0], color='darkblue', linewidth=2.5,
               label='强关联 (NPMI > 0.4)', alpha=0.7),
        Line2D([0], [0], color='gray', linewidth=1.5,
               label='中等关联 (0.3 < NPMI ≤ 0.4)', alpha=0.5),
        Line2D([0], [0], color='lightgray', linewidth=0.8, linestyle='dashed',
               label='弱关联 (NPMI ≤ 0.3)', alpha=0.3),
    ]

    # 添加社区颜色图例
    for comm_id in sorted(community_colors.keys()):
        if comm_id in community_labels:
            label = f"社区{comm_id}: {community_labels[comm_id]}"
        else:
            label = f"社区{comm_id}"

        legend_elements.append(
            Patch(facecolor=community_colors[comm_id],
                  edgecolor='white',
                  label=label)
        )

    plt.legend(handles=legend_elements,
               loc='upper right',
               fontsize=9,
               frameon=True,
               fancybox=True,
               shadow=True,
               bbox_to_anchor=(1.0, 1.0),
               title="图例说明",
               title_fontsize=10)

    # 9. 添加标题和统计信息
    plt.title(f"图4：'{central_node}'意象的自我中心网络\n(边标签显示NPMI关联强度，节点颜色表示语义社区)",
              fontsize=16,
              fontweight='bold',
              pad=25)

    # 统计信息框
    # 计算邻居节点的社区分布
    neighbor_comm_dist = {}
    for neighbor in ego.neighbors(central_node):
        comm_id = G.nodes[neighbor].get('community', 0)
        neighbor_comm_dist[comm_id] = neighbor_comm_dist.get(comm_id, 0) + 1

    # 计算与邻居的平均NPMI
    neighbor_weights = [G[central_node][n]['weight'] for n in ego.neighbors(central_node)]
    avg_npmi = np.mean(neighbor_weights) if neighbor_weights else 0

    # 找出最强关联
    strongest_links = []
    for neighbor in ego.neighbors(central_node):
        weight = G[central_node][neighbor]['weight']
        comm_id = G.nodes[neighbor].get('community', 0)
        strongest_links.append((neighbor, weight, comm_id))

    strongest_links.sort(key=lambda x: x[1], reverse=True)
    top_links = strongest_links[:3]  # 取前3个最强关联

    stats_text = f"""网络统计信息：
• 中心节点度：{ego.degree(central_node)}
• 直接邻居数：{ego.number_of_nodes() - 1}
• 平均关联强度：{avg_npmi:.3f}
• 跨社区连接数：{len(neighbor_comm_dist)}个

邻居社区分布："""

    for comm_id, count in sorted(neighbor_comm_dist.items()):
        comm_name = community_labels.get(comm_id, f"社区{comm_id}")
        stats_text += f"\n  • {comm_name}: {count}个"

    if top_links:
        stats_text += f"\n\n最强关联："
        for neighbor, weight, comm_id in top_links:
            comm_name = community_labels.get(comm_id, f"社区{comm_id}")
            stats_text += f"\n  • {neighbor}: {weight:.3f} ({comm_name})"

    plt.text(0.02, 0.02, stats_text,
             transform=plt.gca().transAxes,
             fontsize=9.5,
             verticalalignment='bottom',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                       edgecolor='orange', linewidth=1.5))

    plt.axis('off')

    # 10. 保存图表
    print("保存图表...")
    output_png = 'figure4_杨柳_自我中心网络.png'
    output_pdf = 'figure4_杨柳_自我中心网络.pdf'

    plt.tight_layout()
    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()

    print(f"\n✓ 图4已生成：")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    # 11. 打印详细分析报告
    print("\n" + "=" * 60)
    print(f"'{central_node}'自我中心网络详细分析报告")
    print("=" * 60)

    print(f"\n一、基本网络特性")
    print(f"   1. 中心节点：'{central_node}'")
    print(
        f"   2. 网络中度：{G.degree(central_node)}（全网排名第{list(sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)).index((central_node, G.degree(central_node))) + 1}）")
    print(f"   3. 直接邻居：{list(ego.neighbors(central_node))}")

    print(f"\n二、语义社区连接分析")
    print(f"   中心节点连接了 {len(neighbor_comm_dist)} 个语义社区：")
    total_neighbors = len(list(ego.neighbors(central_node)))
    for comm_id, count in sorted(neighbor_comm_dist.items()):
        comm_name = community_labels.get(comm_id, f"社区{comm_id}")
        percentage = count / total_neighbors * 100
        print(f"   • {comm_name}：{count}个邻居 ({percentage:.1f}%)")

    print(f"\n三、关联强度分析")
    print(f"   1. 平均NPMI关联强度：{avg_npmi:.3f}")
    print(f"   2. 关联强度分布：")
    print(f"      • 强关联 (NPMI > 0.4)：{len([w for w in neighbor_weights if w > 0.4])}个")
    print(f"      • 中等关联 (0.3-0.4)：{len([w for w in neighbor_weights if 0.3 <= w <= 0.4])}个")
    print(f"      • 弱关联 (NPMI < 0.3)：{len([w for w in neighbor_weights if w < 0.3])}个")

    print(f"\n四、网络角色判断")
    if len(neighbor_comm_dist) >= 3:
        print(f"   ✓ '{central_node}'是典型的桥梁节点")
        print(f"     连接多个语义社区，在唐诗中具有多重象征意义")
    elif len(neighbor_comm_dist) == 2:
        print(f"   • '{central_node}'是连接两个语义社区的中介节点")
        print(f"     主要在这两个主题间建立语义联系")
    else:
        print(f"   • '{central_node}'主要集中在一个语义社区内")
        print(f"     在该主题中扮演核心角色")

    # 计算桥梁系数
    if len(neighbor_comm_dist) > 1:
        bridge_score = len(neighbor_comm_dist) / len(list(ego.neighbors(central_node)))
        print(f"\n五、桥梁系数评估")
        print(f"   桥梁系数：{bridge_score:.3f}（范围0-1，越高表示桥梁作用越强）")
        if bridge_score > 0.4:
            print(f"   → '{central_node}'在唐诗意象系统中具有显著的桥梁功能")

    return ego, pos


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("唐诗意象自我中心网络生成系统")
        print("=" * 60)

        ego, pos = generate_yangliu_ego_network()

        print("\n" + "=" * 60)
        print("图表生成完成！")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n错误：找不到文件 {e.filename}")
        print("请确保以下文件在当前目录：")
        print("1. cleaned_poetry_pairs.csv")
        print("2. community_assignment.csv")
        print("3. community_themes.csv")

    except Exception as e:
        print(f"\n生成图表时出错：{e}")
        import traceback

        traceback.print_exc()