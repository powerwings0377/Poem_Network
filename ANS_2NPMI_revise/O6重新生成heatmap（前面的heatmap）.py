import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
EDGES_FILE = "重新生成Gephi图/cleaned_poetry_pairs.csv"
COMMUNITY_FILE = "重新生成Gephi图/community_assignment.csv"
THEMES_FILE = "重新生成Gephi图/community_themes.csv"

OUTPUT_PNG = "figure4_community_heatmap.png"
OUTPUT_PDF = "figure4_community_heatmap.pdf"

# 可选的社区主题英文翻译（如果 themes 文件没有英文标签，可在此补充）
THEME_TRANS = {
    "花春风年": "Spring Scene & Boudoir Sentiment",
    "秋月风山": "Journey, Homesickness, & Parting Sorrow",
    "人白相年": "Life Experience & Career Aspiration",
    "山水人年": "Reclusive Landscape & Zen Philosophy",
    "千山天万": "Court & History",
    "风秋山客": "Autumn Journey"
}
# ==============================================

def main():
    print("正在读取数据...")
    edges = pd.read_csv(EDGES_FILE)
    comm_df = pd.read_csv(COMMUNITY_FILE)
    themes_df = pd.read_csv(THEMES_FILE)

    # 检查必需列
    required_edges = {"词1", "词2", "NPMI值"}
    required_comm = {"phrase", "community"}
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"边文件缺少列 {required_edges - set(edges.columns)}")
    if not required_comm.issubset(comm_df.columns):
        raise ValueError(f"社区文件缺少列 {required_comm - set(comm_df.columns)}")

    # 构建短语 -> 社区映射
    phrase_to_comm = dict(zip(comm_df["phrase"].astype(str).str.strip(),
                              comm_df["community"]))
    print(f"加载了 {len(phrase_to_comm)} 个节点")

    # 社区列表（排序）
    communities = sorted(set(phrase_to_comm.values()))
    n = len(communities)
    print(f"发现社区: {communities}")

    # 初始化累加矩阵和计数矩阵
    sum_npmi = np.zeros((n, n))
    count = np.zeros((n, n), dtype=int)

    # 遍历每条边，累加 NPMI
    processed = 0
    for _, row in edges.iterrows():
        w1 = str(row["词1"]).strip()
        w2 = str(row["词2"]).strip()
        npmi = row["NPMI值"]

        if w1 in phrase_to_comm and w2 in phrase_to_comm:
            c1 = phrase_to_comm[w1]
            c2 = phrase_to_comm[w2]
            i = communities.index(c1)
            j = communities.index(c2)
            sum_npmi[i, j] += npmi
            sum_npmi[j, i] += npmi   # 对称
            count[i, j] += 1
            count[j, i] += 1
            processed += 1

    print(f"处理了 {processed} 条有效边")

    # 计算平均 NPMI（避免除零）
    avg_npmi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if count[i, j] > 0:
                avg_npmi[i, j] = sum_npmi[i, j] / count[i, j]

    # 读取社区主题（用于标签）
    # 推断社区ID列名
    if "community_id" in themes_df.columns:
        comm_col = "community_id"
    elif "community" in themes_df.columns:
        comm_col = "community"
    else:
        # 若无，则用索引作为社区ID（需要确保顺序）
        comm_col = None

    labels = []
    for comm_id in communities:
        if comm_col is not None:
            row = themes_df[themes_df[comm_col] == comm_id]
            if not row.empty:
                theme_cn = row.iloc[0]["theme_label"]
                # 尝试翻译成英文
                theme_en = THEME_TRANS.get(theme_cn, theme_cn)
                size = row.iloc[0].get("size", "")
                label = f"C{comm_id}\n{theme_en}"
                if size:
                    label += f"\n({size})"
                labels.append(label)
                continue
        # 后备标签
        labels.append(f"C{comm_id}")

    # 绘制热力图
    print("绘制热力图...")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(avg_npmi,
                     annot=True,
                     fmt=".3f",
                     cmap="YlOrRd",
                     linewidths=1,
                     linecolor="white",
                     square=True,
                     xticklabels=labels,
                     yticklabels=labels,
                     cbar_kws={"label": "Average NPMI Association Strength", "shrink": 0.8})
    # 在单元格中添加边数小字
    for i in range(n):
        for j in range(n):
            if count[i, j] > 0:
                ax.text(j + 0.5, i + 0.65, f"n={count[i, j]}",
                        ha="center", va="top", fontsize=8, color="darkblue", fontweight="bold")

    plt.title("Figure 3: Inter-Community Average NPMI Heatmap\n(n = number of connecting edges)",
              fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存图片
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"已保存: {OUTPUT_PNG} 和 {OUTPUT_PDF}")

    # 可选：显示
    plt.show()

    # 输出统计表
    print("\n=== 社区间平均NPMI统计 ===")
    for i, c1 in enumerate(communities):
        for j, c2 in enumerate(communities[i:], i):
            if count[i, j] > 0:
                print(f"C{c1} ↔ C{c2} : avg NPMI = {avg_npmi[i, j]:.4f}  (edges = {count[i, j]})")

if __name__ == "__main__":
    main()