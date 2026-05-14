# generate_figure2_corrected_final.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_network_visualization():
    print("正在生成图2：唐诗诗语网络可视化...")

    # 1. 加载数据
    print("1. 加载数据...")
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    print(f"边数据: {len(edges_df)}条")
    print(f"社区分配: {len(community_df)}个节点")
    print(f"社区主题: {len(themes_df)}个社区")

    # 打印列名以便调试
    print("\n文件列名:")
    print(f"edges_df 列: {list(edges_df.columns)}")
    print(f"community_df 列: {list(community_df.columns)}")
    print(f"themes_df 列: {list(themes_df.columns)}")

    # 2. 创建网络图
    print("\n2. 创建网络图...")
    G = nx.Graph()

    # 创建诗语到社区的映射
    community_dict = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        # 根据您的CSV文件，社区列名是'community'
        comm_id = int(row['community'])  # 0, 1, 2, 3
        community_dict[phrase] = comm_id
        G.add_node(phrase, community=comm_id)

    print(f"已添加 {len(G.nodes())} 个节点")

    # 添加边
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in community_dict and target in community_dict:
            G.add_edge(source, target, weight=weight)
            edge_count += 1

    print(f"已添加 {edge_count} 条边")
    print(f"最终网络: {G.number_of_nodes()}节点, {G.number_of_edges()}边")

    # 3. 读取实际的社区大小和主题信息
    community_sizes = {}
    community_labels = {}

    # 从community_themes.csv读取实际信息
    print("\n3. 读取社区主题信息...")
    for _, row in themes_df.iterrows():
        # 检查列名存在
        if 'community_id' in themes_df.columns:
            comm_id = int(row['community_id'])
        elif 'community' in themes_df.columns:
            comm_id = int(row['community'])
        else:
            # 如果没有找到社区ID列，使用索引
            comm_id = int(row.name)

        size = int(row['size'])
        theme_label = row['theme_label']

        # 处理theme_chars字段
        if 'theme_chars' in themes_df.columns:
            if isinstance(row['theme_chars'], str):
                theme_chars = row['theme_chars'].split()
            else:
                theme_chars = []
        else:
            theme_chars = []

        community_sizes[comm_id] = size
        community_labels[comm_id] = {
            'label': theme_label,
            'chars': theme_chars[:4] if len(theme_chars) >= 4 else theme_chars,
            'size': size
        }

        print(f"  社区{comm_id}: {theme_label} ({size}节点)")

    # 如果community_themes.csv没有数据，使用默认
    if not community_labels:
        print("警告: community_themes.csv为空，使用默认社区信息")
        community_sizes = {0: 213, 1: 169, 2: 191, 3: 95}
        community_labels = {
            0: {'label': '花春风年', 'chars': ['花', '春', '风', '年'], 'size': 213},
            1: {'label': '秋月风山', 'chars': ['秋', '月', '风', '山'], 'size': 169},
            2: {'label': '人白相年', 'chars': ['人', '白', '相', '年'], 'size': 191},
            3: {'label': '山水人年', 'chars': ['山', '水', '人', '年'], 'size': 95}
        }

    # 4. 计算节点大小（基于度中心性）
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # 节点大小：最小200，最大2000
    node_sizes = []
    for node in G.nodes():
        deg = degrees.get(node, 1)
        size = 200 + (deg / max_degree) * 1800
        node_sizes.append(size)

    # 5. 准备节点颜色（按社区）
    # 根据您的社区编号定义颜色
    community_colors = {
        0: '#E74C3C',  # 红色
        1: '#3498DB',  # 蓝色
        2: '#2ECC71',  # 绿色
        3: '#9B59B6'  # 紫色
    }

    node_colors = []
    for node in G.nodes():
        comm_id = G.nodes[node].get('community', 0)
        color = community_colors.get(comm_id, '#95A5A6')
        node_colors.append(color)

    # 6. 创建布局
    print("\n4. 计算网络布局...")

    # 为了提高性能，可以只使用重要的节点进行布局计算
    important_nodes = [node for node, deg in degrees.items() if deg > 30]
    if len(important_nodes) > 100:
        important_nodes = important_nodes[:100]  # 限制在100个重要节点

    if important_nodes:
        G_important = G.subgraph(important_nodes)
        pos_important = nx.spring_layout(G_important, k=2.0, iterations=100, seed=42)

        # 为其他节点分配位置
        pos = {}
        for node in G.nodes():
            if node in pos_important:
                pos[node] = pos_important[node]
            else:
                # 找到邻居的位置平均值
                neighbors = list(G.neighbors(node))
                positioned_neighbors = [n for n in neighbors if n in pos_important]
                if positioned_neighbors:
                    avg_x = np.mean([pos_important[n][0] for n in positioned_neighbors])
                    avg_y = np.mean([pos_important[n][1] for n in positioned_neighbors])
                    # 添加小的随机偏移
                    pos[node] = (avg_x + np.random.uniform(-0.3, 0.3),
                                 avg_y + np.random.uniform(-0.3, 0.3))
                else:
                    pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # 7. 绘制网络图
    print("5. 绘制网络图...")
    fig, ax = plt.subplots(figsize=(16, 14))

    # 绘制边（分批次绘制，避免内存问题）
    print("  绘制边...")
    edges_batch = list(G.edges())[:10000]  # 先绘制前10000条边
    edge_weights = [G[u][v]['weight'] for u, v in edges_batch]

    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            normalized_weights = [(w - min_weight) / (max_weight - min_weight)
                                  for w in edge_weights]
        else:
            normalized_weights = [0.5] * len(edge_weights)

        for (u, v), alpha in zip(edges_batch, normalized_weights):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   alpha=alpha * 0.3 + 0.05,  # 透明度0.05-0.35
                                   width=alpha * 0.8 + 0.2,  # 线宽0.2-1.0
                                   edge_color='gray',
                                   ax=ax)

    # 绘制节点
    print("  绘制节点...")
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85,
                           edgecolors='white',
                           linewidths=1.0,
                           ax=ax)

    # 只标注重要的节点（度高的节点）
    important_nodes_labels = []
    for node, degree in degrees.items():
        if degree > 80:  # 度大于80的节点
            important_nodes_labels.append(node)

    print(f"  标注 {len(important_nodes_labels)} 个重要节点")

    if important_nodes_labels:
        labels = {node: node for node in important_nodes_labels}
        nx.draw_networkx_labels(G, pos,
                                labels=labels,
                                font_size=10,
                                font_weight='bold',
                                ax=ax)

    # 8. 添加图例
    from matplotlib.patches import Patch

    legend_elements = []
    for comm_id in sorted(community_colors.keys()):
        if comm_id in community_labels:
            info = community_labels[comm_id]
            label = f"社区{comm_id}: {info['label']} ({info['size']}节点)"
        else:
            label = f"社区{comm_id} ({community_sizes.get(comm_id, 0)}节点)"

        legend_elements.append(
            Patch(facecolor=community_colors[comm_id],
                  edgecolor='white',
                  label=label)
        )

    ax.legend(handles=legend_elements,
              loc='upper right',
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              bbox_to_anchor=(1.0, 1.0))

    # 9. 添加标题和说明
    plt.title('图2：唐诗诗语语义网络可视化\n(节点按社区着色，大小与度中心性成正比)',
              fontsize=18,
              fontweight='bold',
              pad=20)

    # 10. 在角落添加网络统计信息
    stats_text = f"""网络统计：
• 节点数: {G.number_of_nodes()}
• 边数: {G.number_of_edges()}
• 网络密度: {nx.density(G):.4f}
• 平均度: {np.mean(list(degrees.values())):.1f}
• 模块度: 0.183"""

    plt.text(0.02, 0.02, stats_text,
             transform=ax.transAxes,
             fontsize=11,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.axis('off')

    # 11. 保存图表
    plt.tight_layout()
    output_png = 'figure2_poem_network.png'
    output_pdf = 'figure2_poem_network.pdf'

    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()
    print(f"\n✓ 图2已生成:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    # 12. 额外生成一个简化版网络图（只显示重要节点）
    print("\n6. 生成简化版网络图（只显示度>50的节点）...")
    generate_simplified_network(G, degrees, community_colors, community_labels)

    return G, pos, community_labels


def generate_simplified_network(G, degrees, community_colors, community_labels):
    """生成一个简化版的网络图，只显示重要节点"""
    # 只选择度大于50的节点
    important_nodes = [node for node, deg in degrees.items() if deg > 50]

    if len(important_nodes) < 10:
        print("  重要节点太少，跳过简化版")
        return

    G_simple = G.subgraph(important_nodes)

    # 计算布局
    pos = nx.spring_layout(G_simple, k=1.5, iterations=100, seed=42)

    # 准备节点颜色和大小
    node_colors = []
    node_sizes = []
    for node in G_simple.nodes():
        comm_id = G.nodes[node].get('community', 0)
        color = community_colors.get(comm_id, '#95A5A6')
        node_colors.append(color)
        node_sizes.append(degrees[node] * 15 + 100)

    # 绘制
    plt.figure(figsize=(12, 10))

    # 绘制边
    nx.draw_networkx_edges(G_simple, pos,
                           alpha=0.15,
                           width=0.5,
                           edge_color='gray')

    # 绘制节点
    nx.draw_networkx_nodes(G_simple, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors='white',
                           linewidths=1.5)

    # 标注所有节点
    labels = {node: node for node in G_simple.nodes()}
    nx.draw_networkx_labels(G_simple, pos,
                            labels=labels,
                            font_size=9,
                            font_weight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = []
    for comm_id in sorted(community_colors.keys()):
        if comm_id in community_labels:
            info = community_labels[comm_id]
            label = f"社区{comm_id}: {info['label']}"
        else:
            label = f"社区{comm_id}"

        legend_elements.append(
            Patch(facecolor=community_colors[comm_id],
                  edgecolor='white',
                  label=label)
        )

    plt.legend(handles=legend_elements,
               loc='upper right',
               fontsize=9)

    plt.title('图2（简化版）: 唐诗核心诗语网络\n(仅显示度>50的重要节点)',
              fontsize=16,
              fontweight='bold',
              pad=20)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig('figure2_simplified.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')

    plt.show()
    print("  ✓ 简化版已保存为 figure2_simplified.png")


if __name__ == '__main__':
    try:
        G, pos, community_labels = create_network_visualization()
    except Exception as e:
        print(f"\n✗ 生成图表时出错: {e}")
        print("请检查以下文件是否存在且格式正确:")
        print("1. cleaned_poetry_pairs.csv")
        print("2. community_assignment.csv")
        print("3. community_themes.csv")
        print("\n如果文件存在，请检查列名是否正确。")