import pandas as pd
import numpy as np
import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from community import community_louvain
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. 读取数据
def load_and_preprocess(file_path):
    """加载数据并进行初步清洗"""
    df = pd.read_csv(file_path)

    print("原始数据形状:", df.shape)
    print("数据示例:")
    print(df.head())

    # 数据基本信息
    print("\n=== 数据统计信息 ===")
    print(f"总词对数量: {len(df)}")
    print(f"唯一词1数量: {df['词1'].nunique()}")
    print(f"唯一词2数量: {df['词2'].nunique()}")
    print(f"NPMI值范围: [{df['NPMI值'].min():.3f}, {df['NPMI值'].max():.3f}]")
    print(f"PMI值范围: [{df['PMI值'].min():.3f}, {df['PMI值'].max():.3f}]")

    return df


# 2. 数据清洗函数
def clean_data(df):
    """清洗数据：去除重复和字头字尾重叠的词对"""
    original_len = len(df)

    # 创建副本进行操作
    df_clean = df.copy()

    # 规则1：去除词1和词2有字头字尾重叠的情况
    def has_overlap(phrase1, phrase2):
        """检查两个词是否有字头字尾重叠"""
        # 情况1：词1的尾字 = 词2的首字
        if phrase1[-1] == phrase2[0]:
            return True
        # 情况2：词2的尾字 = 词1的首字
        if phrase2[-1] == phrase1[0]:
            return True
        # 情况3：完全包含（如"山高"和"高山"）
        if phrase1 == phrase2[::-1]:
            return True
        return False

    # 应用规则1
    mask = df_clean.apply(lambda row: not has_overlap(row['词1'], row['词2']), axis=1)
    df_clean = df_clean[mask].copy()
    removed_overlap = original_len - len(df_clean)

    # 规则2：去除重复的词对（A,B 和 B,A视为相同）
    def sort_phrases(phrases):
        """将词对按字母顺序排序"""
        a, b = phrases
        return tuple(sorted([a, b]))

    df_clean['sorted_pair'] = df_clean[['词1', '词2']].apply(sort_phrases, axis=1)
    df_clean = df_clean.drop_duplicates(subset='sorted_pair').copy()
    df_clean = df_clean.drop(columns=['sorted_pair'])
    removed_duplicates = original_len - removed_overlap - len(df_clean)

    # 规则3：过滤低相关性的词对（NPMI阈值）
    threshold = 0.3  # 可以根据数据分布调整
    df_clean = df_clean[df_clean['NPMI值'] >= threshold].copy()
    removed_low_corr = original_len - removed_overlap - removed_duplicates - len(df_clean)

    print("\n=== 清洗结果 ===")
    print(f"原始词对数量: {original_len}")
    print(f"去除字头字尾重叠: {removed_overlap} 对")
    print(f"去除重复词对: {removed_duplicates} 对")
    print(f"去除低相关性(NPMI<{threshold}): {removed_low_corr} 对")
    print(f"最终保留词对: {len(df_clean)} 对")
    print(f"清洗保留比例: {len(df_clean) / original_len * 100:.1f}%")

    # 分析剩余数据的NPMI分布
    print("\n=== 清洗后NPMI分布 ===")
    print(f"NPMI均值: {df_clean['NPMI值'].mean():.3f}")
    print(f"NPMI标准差: {df_clean['NPMI值'].std():.3f}")
    print(f"NPMI中位数: {df_clean['NPMI值'].median():.3f}")

    return df_clean


# 3. 选择合适的关联度量
def analyze_correlation_metrics(df):
    """分析不同关联度量的特性"""

    print("\n=== 关联度量分析 ===")

    # 计算各指标的相关性
    correlation_matrix = df[['NPMI值', 'PMI值', '联合概率', '共现次数']].corr()
    print("各指标相关系数矩阵:")
    print(correlation_matrix)

    # NPMI vs PMI 分析
    print(f"\nNPMI-PMI相关系数: {df['NPMI值'].corr(df['PMI值']):.3f}")

    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # NPMI分布
    axes[0, 0].hist(df['NPMI值'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['NPMI值'].mean(), color='red', linestyle='--',
                       label=f'均值={df["NPMI值"].mean():.3f}')
    axes[0, 0].set_xlabel('NPMI值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('NPMI值分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PMI分布
    axes[0, 1].hist(df['PMI值'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(df['PMI值'].mean(), color='red', linestyle='--',
                       label=f'均值={df["PMI值"].mean():.3f}')
    axes[0, 1].set_xlabel('PMI值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('PMI值分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 联合概率分布
    axes[1, 0].hist(df['联合概率'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(df['联合概率'].mean(), color='red', linestyle='--',
                       label=f'均值={df["联合概率"].mean():.6f}')
    axes[1, 0].set_xlabel('联合概率')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('联合概率分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # NPMI vs 共现次数散点图
    scatter = axes[1, 1].scatter(df['共现次数'], df['NPMI值'],
                                 c=df['联合概率'], cmap='viridis',
                                 alpha=0.6, s=30)
    axes[1, 1].set_xlabel('共现次数')
    axes[1, 1].set_ylabel('NPMI值')
    axes[1, 1].set_title('NPMI vs 共现次数')
    plt.colorbar(scatter, ax=axes[1, 1], label='联合概率')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 度量选择建议
    print("\n=== 关联度量选择建议 ===")
    print("1. NPMI值（推荐作为网络权重）：")
    print("   - 优点：归一化到[-1,1]，可比性强")
    print("   - 优点：考虑了共现概率的归一化")
    print("   - 适合：捕捉诗语的语义关联强度")

    print("\n2. PMI值：")
    print("   - 缺点：受边缘概率影响大，值范围不稳定")
    print("   - 不适合：作为网络权重")

    print("\n3. 联合概率：")
    print("   - 缺点：偏向高频词对，不反映特异性关联")
    print("   - 适合：作为权重补充，但不适合单独使用")

    print("\n4. 共现次数：")
    print("   - 缺点：过度依赖词频")
    print("   - 适合：作为筛选阈值，但不适合作为权重")

    return 'NPMI值'  # 推荐使用NPMI作为权重


def build_and_analyze_network(df, weight_column='NPMI值'):
    """构建网络并进行基础分析"""

    print(f"\n=== 使用'{weight_column}'作为边权重构建网络 ===")

    # 创建无向加权图
    G = nx.Graph()

    # 添加节点和边
    for _, row in df.iterrows():
        phrase1 = row['词1']
        phrase2 = row['词2']
        weight = row[weight_column]

        # 添加节点（如果不存在）
        G.add_node(phrase1,
                   frequency=row['词1出现次数'],
                   probability=row['词1概率'])
        G.add_node(phrase2,
                   frequency=row['词2出现次数'],
                   probability=row['词2概率'])

        # 添加边
        G.add_edge(phrase1, phrase2,
                   weight=weight,
                   cooccurrence=row['共现次数'],
                   joint_prob=row['联合概率'])

    # 基础网络统计
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"网络密度: {nx.density(G):.4f}")

    # 计算连通分量
    connected_components = list(nx.connected_components(G))
    print(f"连通分量数量: {len(connected_components)}")

    # 最大连通分量分析
    largest_cc = max(connected_components, key=len)
    G_largest = G.subgraph(largest_cc)
    print(f"最大连通分量大小: {len(largest_cc)} 节点")
    print(f"最大连通分量边数: {G_largest.number_of_edges()}")

    # 度分布分析
    degrees = [d for _, d in G.degree()]
    print(f"平均度: {np.mean(degrees):.2f}")
    print(f"最大度: {max(degrees)}")
    print(f"度标准差: {np.std(degrees):.2f}")

    # 权重分析
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    print(f"边权重均值: {np.mean(weights):.3f}")
    print(f"边权重标准差: {np.std(weights):.3f}")

    return G, G_largest

#4网络构建与基础分析
def calculate_node_centralities(G):
    """计算各种中心性指标"""

    print("\n=== 节点中心性分析 ===")

    # 1. 度数中心性
    degree_centrality = nx.degree_centrality(G)

    # 2. 介数中心性（Betweenness Centrality）
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

    # 3. 接近中心性（Closeness Centrality）
    closeness_centrality = nx.closeness_centrality(G, distance='weight')

    # 4. 特征向量中心性
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

    # 5. PageRank
    pagerank = nx.pagerank(G, weight='weight')

    # 创建节点属性DataFrame
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            'phrase': node,
            'degree': G.degree(node),
            'degree_centrality': degree_centrality[node],
            'betweenness': betweenness_centrality[node],
            'closeness': closeness_centrality[node],
            'eigenvector': eigenvector_centrality[node],
            'pagerank': pagerank[node],
            'frequency': G.nodes[node].get('frequency', 0),
            'probability': G.nodes[node].get('probability', 0)
        })

    nodes_df = pd.DataFrame(nodes_data)

    # 分析不同中心性的相关性
    centrality_cols = ['degree_centrality', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
    centrality_corr = nodes_df[centrality_cols].corr()

    print("中心性指标相关性矩阵:")
    print(centrality_corr)

    # 找出最重要的节点（综合排名）
    for col in centrality_cols:
        nodes_df[f'{col}_rank'] = nodes_df[col].rank(ascending=False)

    nodes_df['avg_rank'] = nodes_df[[f'{col}_rank' for col in centrality_cols]].mean(axis=1)
    nodes_df = nodes_df.sort_values('avg_rank')

    print("\n=== 综合排名TOP 20诗语 ===")
    top_20 = nodes_df.head(20)[['phrase', 'degree', 'degree_centrality',
                                'betweenness', 'avg_rank', 'frequency']]
    print(top_20.to_string(index=False))

    return nodes_df

#5社区发现与主题识别
def detect_communities(G, resolution=1.0):
    """使用Louvain算法进行社区发现"""

    print("\n=== 社区发现分析 ===")

    # 转换为无向图（Louvain需要无向图）
    G_undirected = G.to_undirected()

    # 运行Louvain算法
    partition = community_louvain.best_partition(G_undirected, weight='weight',
                                                 resolution=resolution, random_state=42)

    # 计算模块度
    modularity = community_louvain.modularity(partition, G_undirected, weight='weight')
    print(f"模块度 (Modularity): {modularity:.4f}")
    print(f"发现社区数量: {len(set(partition.values()))}")

    # 统计社区信息
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 社区大小分布
    comm_sizes = [len(nodes) for nodes in communities.values()]
    print(f"社区平均大小: {np.mean(comm_sizes):.1f}")
    print(f"最大社区大小: {max(comm_sizes)}")
    print(f"最小社区大小: {min(comm_sizes)}")

    # 为每个社区自动生成主题标签
    def extract_theme_keywords(phrases, top_k=5):
        """从社区词集中提取主题关键词"""
        # 统计所有字符
        all_chars = []
        for phrase in phrases:
            all_chars.extend(list(phrase))

        from collections import Counter
        char_freq = Counter(all_chars)

        # 过滤掉无意义的虚词
        stop_chars = {'不', '无', '有', '在', '何', '谁', '几', '处', '时', '日'}
        meaningful_chars = [(char, freq) for char, freq in char_freq.items()
                            if char not in stop_chars]

        # 按频率排序
        meaningful_chars.sort(key=lambda x: x[1], reverse=True)

        # 提取高频字
        theme_chars = [char for char, freq in meaningful_chars[:top_k]]
        return theme_chars

    # 生成社区主题
    community_themes = {}
    for comm_id, phrases in communities.items():
        theme_chars = extract_theme_keywords(phrases, top_k=4)
        community_themes[comm_id] = {
            'phrases': phrases,
            'size': len(phrases),
            'theme_chars': theme_chars,
            'theme_label': ''.join(theme_chars)
        }

    # 打印社区信息
    print("\n=== 社区详细信息 ===")
    sorted_communities = sorted(community_themes.items(),
                                key=lambda x: x[1]['size'], reverse=True)

    for comm_id, info in sorted_communities:
        print(f"\n社区 {comm_id} (大小: {info['size']}):")
        print(f"  主题特征字: {info['theme_chars']}")
        print(f"  自动主题标签: {info['theme_label']}")

        # 显示社区内最重要的节点
        subgraph = G.subgraph(info['phrases'])
        if subgraph.number_of_nodes() > 0:
            degree_in_comm = dict(subgraph.degree())
            top_nodes = sorted(degree_in_comm.items(), key=lambda x: x[1], reverse=True)[:8]
            top_phrases = [f"{phrase}({deg})" for phrase, deg in top_nodes]
            print(f"  核心诗语: {', '.join(top_phrases)}")

    return partition, community_themes, modularity


# 新增：导出所有社区代表词（全量词，按重要性排序，仅生成1个汇总CSV）
def export_community_all_sorted_words(G, partition, nodes_df, save_path='community_all_sorted_words.csv'):
    """
    导出所有社区的全量诗语到**单个CSV文件**，按「社区ID升序+社区内重要性降序」排列
    排序核心规则：社区内度数（主）→ 全局综合排名（次），兼顾社区内核心度和全局重要性
    :param G: 最大连通分量网络G_largest（核心分析网络）
    :param partition: 节点-社区ID映射（detect_communities的输出）
    :param nodes_df: 含中心性/综合排名的节点数据（calculate_node_centralities的输出）
    :param save_path: 汇总CSV的保存路径/文件名，默认当前目录
    :return: 排序后的汇总DataFrame（方便后续二次使用）
    """
    print("\n=== 导出所有社区代表词（全量，单CSV，按重要性排序）===")

    # 1. 整理社区-节点映射，仅保留分析网络G中的有效节点
    comm2nodes = defaultdict(list)
    for node in G.nodes():
        comm_id = partition[node]
        comm2nodes[comm_id].append(node)

    # 2. 遍历每个社区，计算节点重要性指标
    all_comm_data = []
    for comm_id, node_list in comm2nodes.items():
        # 构建社区子图，计算**社区内度数**（核心：代表词在本社区的关联强度）
        comm_subG = G.subgraph(node_list)
        inner_degree = dict(comm_subG.degree())

        # 提取每个节点的重要性指标
        for phrase in node_list:
            # 全局综合排名（avg_rank越小，全局越重要，原有代码已计算）
            global_avg_rank = nodes_df[nodes_df['phrase'] == phrase]['avg_rank'].values[0]
            # 全局度数（辅助参考：该词在整个网络的关联数）
            global_deg = G.degree(phrase)
            # 社区内排名（后续统一计算，这里先存核心指标）
            all_comm_data.append({
                '社区ID': comm_id,
                '诗语': phrase,
                '社区内度数': inner_degree[phrase],  # 主排序指标
                '全局度数': global_deg,  # 辅助指标
                '全局综合排名': round(global_avg_rank, 2),  # 次排序指标
                '社区内重要性得分': inner_degree[phrase] + (1000 / global_avg_rank)  # 排序辅助分
            })

    # 3. 转换为DataFrame，按规则排序：社区ID升序 → 社区内度数降序 → 全局综合排名升序
    result_df = pd.DataFrame(all_comm_data)
    result_df = result_df.sort_values(
        by=['社区ID', '社区内度数', '全局综合排名'],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # 4. 为每个社区单独计算「社区内排名」（1为社区核心代表词，数字越小越重要）
    result_df['社区内排名'] = result_df.groupby('社区ID').cumcount() + 1

    # 5. 保存为单个CSV，utf-8-sig编码（Excel打开无中文乱码，核心！）
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    # 6. 控制台输出统计信息，方便核对
    total_community = len(comm2nodes)
    total_phrase = len(result_df)
    print(f"✓ 导出成功！共{total_community}个社区，{total_phrase}个诗语")
    print(f"✓ 汇总CSV文件保存至：{save_path}")
    print(f"✓ 排序规则：社区ID升序 → 社区内度数（降）→ 全局综合排名（升）")
    print(f"✓ 核心列说明：社区内排名（1=社区核心代表词）、社区内度数（本社区关联强度）")

    # 可选：输出前3大社区的核心信息，快速预览
    top3_comm = result_df.groupby('社区ID').size().nlargest(3)
    print(f"\n📌 前3大社区预览：")
    for comm_id, size in top3_comm.items():
        core_phrase = result_df[result_df['社区ID'] == comm_id]['诗语'].iloc[0]
        print(f"   社区{comm_id}：{size}个诗语，核心代表词→{core_phrase}")

    return result_df

def visualize_network(G, partition, nodes_df, top_n=50):
    """网络可视化"""

    print("\n=== 生成网络可视化 ===")

    # 1. 整体网络布局（使用最大连通分量）
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc)

    # 创建布局
    pos = nx.spring_layout(G_main, k=2, iterations=50, seed=42, weight='weight')

    # 节点大小基于度数
    node_sizes = [G_main.degree(n) * 20 + 50 for n in G_main.nodes()]

    # 节点颜色基于社区
    community_colors = {}
    unique_communities = set(partition.values())
    cmap = plt.cm.Set3
    for i, comm in enumerate(unique_communities):
        community_colors[comm] = cmap(i / max(1, len(unique_communities) - 1))

    node_colors = [community_colors[partition[n]] for n in G_main.nodes()]

    # 绘制网络
    plt.figure(figsize=(16, 12))

    # 绘制边（透明度基于权重）
    edges = G_main.edges()
    weights = [G_main[u][v]['weight'] for u, v in edges]
    normalized_weights = [(w - min(weights)) / (max(weights) - min(weights))
                          for w in weights]

    for (u, v), alpha in zip(edges, normalized_weights):
        nx.draw_networkx_edges(G_main, pos, edgelist=[(u, v)],
                               alpha=alpha * 0.5 + 0.1, width=alpha * 0.5 + 0.1,
                               edge_color='gray')

    # 绘制节点
    nx.draw_networkx_nodes(G_main, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.8,
                           edgecolors='black', linewidths=0.5)

    # 只标注重要的节点
    important_nodes = []
    for node in G_main.nodes():
        if nodes_df[nodes_df['phrase'] == node]['avg_rank'].values[0] <= top_n:
            important_nodes.append(node)

    labels = {node: node for node in important_nodes}
    nx.draw_networkx_labels(G_main, pos, labels, font_size=9,
                            font_family='SimHei', font_weight='bold')

    plt.title(f'唐诗诗语关联网络（N={len(G_main.nodes())}, E={len(G_main.edges())}）',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('poetry_network.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 中心性分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 度分布直方图
    degrees = [d for _, d in G_main.degree()]
    axes[0, 0].hist(degrees, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('度数')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('节点度分布')
    axes[0, 0].grid(True, alpha=0.3)

    # 权重分布
    weights = [G_main[u][v]['weight'] for u, v in G_main.edges()]
    axes[0, 1].hist(weights, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('边权重 (NPMI)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('边权重分布')
    axes[0, 1].grid(True, alpha=0.3)

    # 社区大小分布
    community_sizes = []
    for comm_id in set(partition.values()):
        size = sum(1 for node in G_main.nodes() if partition[node] == comm_id)
        if size > 0:
            community_sizes.append(size)

    axes[1, 0].bar(range(len(community_sizes)), sorted(community_sizes, reverse=True),
                   alpha=0.7, color='mediumseagreen')
    axes[1, 0].set_xlabel('社区排名')
    axes[1, 0].set_ylabel('社区大小')
    axes[1, 0].set_title('社区大小分布')
    axes[1, 0].grid(True, alpha=0.3)

    # 中心性相关性热图
    centrality_cols = ['degree_centrality', 'betweenness', 'closeness', 'eigenvector']
    corr_matrix = nodes_df[nodes_df['phrase'].isin(G_main.nodes())][centrality_cols].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(centrality_cols)))
    axes[1, 1].set_yticks(range(len(centrality_cols)))
    axes[1, 1].set_xticklabels([col.replace('_', '\n') for col in centrality_cols])
    axes[1, 1].set_yticklabels([col.replace('_', '\n') for col in centrality_cols])
    axes[1, 1].set_title('中心性指标相关性')

    # 添加相关系数值
    for i in range(len(centrality_cols)):
        for j in range(len(centrality_cols)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig('network_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 社区结构可视化
    visualize_community_structure(G_main, partition, community_colors)

#6. 可视化展示
def visualize_community_structure(G, partition, community_colors):
    """社区结构可视化"""

    # 选择最大的几个社区进行详细展示
    community_sizes = {}
    for node, comm_id in partition.items():
        if node in G.nodes():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

    # 选择前6大社区
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (comm_id, size) in enumerate(top_communities):
        if idx >= 6:
            break

        # 提取该社区的节点
        comm_nodes = [n for n in G.nodes() if partition[n] == comm_id]
        if len(comm_nodes) < 2:
            continue

        # 创建社区子图
        G_comm = G.subgraph(comm_nodes)

        # 社区内部布局
        pos = nx.spring_layout(G_comm, seed=42)

        # 绘制社区网络
        nx.draw_networkx_nodes(G_comm, pos, ax=axes[idx],
                               node_size=100,
                               node_color=[community_colors[comm_id]],
                               alpha=0.8)
        nx.draw_networkx_edges(G_comm, pos, ax=axes[idx],
                               alpha=0.3, width=0.5)

        # 标注节点
        if len(comm_nodes) <= 15:
            nx.draw_networkx_labels(G_comm, pos, ax=axes[idx],
                                    font_size=8, font_family='SimHei')

        axes[idx].set_title(f'社区{comm_id} (大小: {size})', fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('主要社区内部结构', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('communities_structure.png', dpi=300, bbox_inches='tight')
    plt.show()

#7主要函数
def main_analysis_pipeline(file_path):
    """主分析流程"""

    print("=" * 60)
    print("唐诗诗语网络分析系统")
    print("=" * 60)

    # 步骤1: 加载数据
    print("\n[步骤1] 加载数据")
    df = load_and_preprocess(file_path)

    # 步骤2: 数据清洗
    print("\n[步骤2] 数据清洗")
    df_clean = clean_data(df)

    # 步骤3: 关联度量分析
    print("\n[步骤3] 关联度量分析")
    weight_column = analyze_correlation_metrics(df_clean)

    # 步骤4: 网络构建
    print("\n[步骤4] 网络构建与分析")
    G, G_largest = build_and_analyze_network(df_clean, weight_column)

    # 步骤5: 中心性分析
    nodes_df = calculate_node_centralities(G_largest)

    # 步骤6: 社区发现
    partition, community_themes, modularity = detect_communities(G_largest)
    # 新增步骤：导出所有社区全量代表词（单CSV，按重要性排序）
    comm_sorted_df = export_community_all_sorted_words(G_largest, partition, nodes_df)


    # 步骤7: 可视化
    visualize_network(G_largest, partition, nodes_df)

    # 保存结果
    print("\n[步骤8] 保存分析结果")

    # 保存清洗后的数据
    df_clean.to_csv('cleaned_poetry_pairs.csv', index=False, encoding='utf-8-sig')

    # 保存节点中心性数据
    nodes_df.to_csv('node_centralities.csv', index=False, encoding='utf-8-sig')

    # 保存社区分配
    partition_df = pd.DataFrame([(node, comm_id) for node, comm_id in partition.items()],
                                columns=['phrase', 'community'])
    partition_df.to_csv('community_assignment.csv', index=False, encoding='utf-8-sig')

    # 保存社区主题信息
    themes_data = []
    for comm_id, info in community_themes.items():
        themes_data.append({
            'community_id': comm_id,
            'size': info['size'],
            'theme_chars': ' '.join(info['theme_chars']),
            'theme_label': info['theme_label'],
            'phrases': ' '.join(info['phrases'][:50])  # 只保存前20个
        })
    themes_df = pd.DataFrame(themes_data)
    themes_df.to_csv('community_themes.csv', index=False, encoding='utf-8-sig')

    print("\n✓ 所有分析完成！")
    print(f"✓ 清洗后数据保存至: cleaned_poetry_pairs.csv")
    print(f"✓ 节点中心性保存至: node_centralities.csv")
    print(f"✓ 社区分配保存至: community_assignment.csv")
    print(f"✓ 社区主题保存至: community_themes.csv")

    return {
        'df_clean': df_clean,
        'G': G,
        'G_largest': G_largest,
        'nodes_df': nodes_df,
        'partition': partition,
        'community_themes': community_themes
    }


# 运行完整分析流程
if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = "output_npmi_20260201_210505_npmi.csv"

    try:
        results = main_analysis_pipeline(file_path)

        # 输出重要发现摘要
        print("\n" + "=" * 60)
        print("重要发现摘要")
        print("=" * 60)

        G_largest = results['G_largest']
        nodes_df = results['nodes_df']
        community_themes = results['community_themes']

        # 网络特性
        print(f"1. 网络规模: {G_largest.number_of_nodes()} 节点, {G_largest.number_of_edges()} 边")
        print(f"2. 网络密度: {nx.density(G_largest):.4f}")
        print(f"3. 平均聚类系数: {nx.average_clustering(G_largest):.3f}")

        # 核心诗语
        top_core = nodes_df.head(5)['phrase'].tolist()
        print(f"4. 核心诗语 (综合排名前5): {', '.join(top_core)}")

        # 社区发现
        print(f"5. 发现社区数量: {len(community_themes)}")
        print(f"6. 模块度: {community_louvain.modularity(results['partition'], G_largest, weight='weight'):.3f}")

        # 主要主题
        print("7. 主要诗语主题:")
        sorted_themes = sorted(community_themes.items(),
                               key=lambda x: x[1]['size'], reverse=True)[:5]
        for comm_id, info in sorted_themes:
            print(f"   - 主题{comm_id}: {info['theme_label']} ({info['size']}个诗语)")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        print("请确保文件路径正确，并包含以下列：")
        print("词1,词2,NPMI值,PMI值,共现次数,词1出现次数,词2出现次数,词1概率,词2概率,联合概率")
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")