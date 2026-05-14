# generate_figure3_corrected_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_community_heatmap():
    print("正在生成图3：社区间关联强度热力图...")

    # 1. 加载数据
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    # 打印列名以便调试
    print("文件列名:")
    print(f"edges_df: {list(edges_df.columns)}")
    print(f"community_df: {list(community_df.columns)}")
    print(f"themes_df: {list(themes_df.columns)}")

    # 创建诗语到社区的映射
    phrase_to_community = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        # 根据您的文件，列名是'community'
        comm_id = int(row['community'])
        phrase_to_community[phrase] = comm_id

    # 2. 从themes_df获取实际的社区信息
    community_info = {}
    for _, row in themes_df.iterrows():
        # 检查列名
        if 'community_id' in themes_df.columns:
            comm_id = int(row['community_id'])
        elif 'community' in themes_df.columns:
            comm_id = int(row['community'])
        else:
            # 使用索引作为社区ID
            comm_id = int(row.name)

        theme_label = str(row['theme_label'])

        # 处理size字段
        if 'size' in themes_df.columns:
            size = int(row['size'])
        else:
            size = 0

        # 处理theme_chars字段
        if 'theme_chars' in themes_df.columns and pd.notna(row['theme_chars']):
            theme_chars = str(row['theme_chars'])
        else:
            theme_chars = ""

        community_info[comm_id] = {
            'label': theme_label,
            'chars': theme_chars,
            'size': size
        }

    print(f"\n社区信息:")
    for comm_id, info in sorted(community_info.items()):
        print(f"  社区{comm_id}: {info['label']} ({info['size']}节点)")

    # 3. 计算社区间的平均NPMI值
    communities = sorted(set(phrase_to_community.values()))
    n_communities = len(communities)

    print(f"\n检测到 {n_communities} 个社区: {communities}")

    # 初始化矩阵
    heatmap_matrix = np.zeros((n_communities, n_communities))
    count_matrix = np.zeros((n_communities, n_communities), dtype=int)

    # 遍历所有边，累加权重和计数
    print("计算社区间关联...")
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in phrase_to_community and target in phrase_to_community:
            comm1 = phrase_to_community[source]
            comm2 = phrase_to_community[target]

            # 转换为矩阵索引
            idx1 = communities.index(comm1)
            idx2 = communities.index(comm2)

            heatmap_matrix[idx1, idx2] += weight
            heatmap_matrix[idx2, idx1] += weight  # 对称矩阵
            count_matrix[idx1, idx2] += 1
            count_matrix[idx2, idx1] += 1
            edge_count += 1

            if edge_count % 5000 == 0:
                print(f"  已处理 {edge_count} 条边...")

    print(f"  总共处理 {edge_count} 条边")

    # 计算平均值（避免除以0）
    for i in range(n_communities):
        for j in range(n_communities):
            if count_matrix[i, j] > 0:
                heatmap_matrix[i, j] = heatmap_matrix[i, j] / count_matrix[i, j]
            else:
                heatmap_matrix[i, j] = 0

    # 4. 创建热力图
    print("\n生成热力图...")
    plt.figure(figsize=(10, 8))

    # 生成x轴和y轴标签
    x_labels = []
    y_labels = []
    for comm_id in communities:
        info = community_info.get(comm_id, {})
        label = info.get('label', f'社区{comm_id}')
        size = info.get('size', 0)
        x_labels.append(f"{label}\n(社区{comm_id})")
        y_labels.append(f"{label}\n(社区{comm_id})")

    # 使用seaborn的热力图
    ax = sns.heatmap(heatmap_matrix,
                     annot=True,
                     fmt=".3f",
                     cmap='YlOrRd',
                     linewidths=1,
                     linecolor='white',
                     square=True,
                     xticklabels=x_labels,
                     yticklabels=y_labels,
                     cbar_kws={'label': '平均NPMI关联强度', 'shrink': 0.8})

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)

    # 5. 在单元格中添加连接数量（小字）
    for i in range(n_communities):
        for j in range(n_communities):
            if count_matrix[i, j] > 0:
                ax.text(j + 0.5, i + 0.65,
                        f'n={count_matrix[i, j]}',
                        ha='center',
                        va='top',
                        fontsize=8,
                        color='darkblue',
                        fontweight='bold')

    # 6. 添加标题
    plt.title('图3：社区间平均NPMI关联强度热力图\n(n为连接边数)',
              fontsize=16,
              fontweight='bold',
              pad=20)

    plt.tight_layout()

    # 7. 保存图表
    output_png = 'figure3_community_heatmap.png'
    output_pdf = 'figure3_community_heatmap.pdf'

    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()

    # 8. 打印详细统计信息
    print("\n=== 社区间关联统计详情 ===")
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities[i:], i):  # 只显示上三角，避免重复
            if count_matrix[i, j] > 0:
                label1 = community_info.get(comm1, {}).get('label', f'社区{comm1}')
                label2 = community_info.get(comm2, {}).get('label', f'社区{comm2}')

                print(f"\n{label1} (社区{comm1}) ↔ {label2} (社区{comm2}):")
                print(f"  平均NPMI: {heatmap_matrix[i, j]:.4f}")
                print(f"  连接边数: {count_matrix[i, j]}")
                if i != j:
                    proportion = count_matrix[i, j] / len(edges_df) * 100
                    print(f"  占总边比例: {proportion:.2f}%")

    print(f"\n✓ 图3已生成:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    return heatmap_matrix, count_matrix, community_info


if __name__ == '__main__':
    try:
        heatmap_matrix, count_matrix, community_info = create_community_heatmap()
    except Exception as e:
        print(f"\n✗ 生成图表时出错: {e}")
        import traceback

        traceback.print_exc()