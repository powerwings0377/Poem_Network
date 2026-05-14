# generate_legend_only_english.py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========== 用户配置（请按实际修改） ==========
# 社区颜色（你指定的）
community_colors = {
    0: '#D37CCF',
    1: '#68AE36',
    2: '#FF753E',
    3: '#00B8DB'
}

# 英文标签（主题 + 节点数，参考原 program 中的翻译）
# 请根据你的实际节点数修改 size
community_labels = {
    0: 'Community 0: Spring Scene & Boudoir Sentiment (213 nodes)',
    1: 'Community 1: Autumn Journey & Homesickness (169 nodes)',
    2: 'Community 2: Life Experience & Career Aspiration (191 nodes)',
    3: 'Community 3: Reclusive Landscape & Zen Philosophy (95 nodes)'
}

# 图例样式（与原 generate_simplified_network 完全一致）
LEGEND_FONTSIZE = 9
LEGEND_LOC = 'center'          # 图例居中，方便裁剪
FRAMEON = True
FANCYBOX = True
SHADOW = True
EDGECOLOR = 'white'            # 色块边缘白色

# 输出文件
OUTPUT_PNG = 'legend_figure5_english.png'
OUTPUT_PDF = 'legend_figure5_english.pdf'

# ========== 生成纯图例图片 ==========
fig, ax = plt.subplots(figsize=(5, 3.2))  # 尺寸根据标签长度调整
ax.axis('off')

# 构建图例句柄
legend_elements = []
for comm_id in sorted(community_colors.keys()):
    legend_elements.append(
        Patch(facecolor=community_colors[comm_id],
              edgecolor=EDGECOLOR,
              label=community_labels[comm_id])
    )

# 添加图例，使用和原图完全相同的参数
legend = ax.legend(handles=legend_elements,
                   loc=LEGEND_LOC,
                   fontsize=LEGEND_FONTSIZE,
                   frameon=FRAMEON,
                   fancybox=FANCYBOX,
                   shadow=SHADOW)

# 保存图片（白色背景，分辨率300）
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
plt.show()

print(f"图例已生成：{OUTPUT_PNG} 和 {OUTPUT_PDF}")