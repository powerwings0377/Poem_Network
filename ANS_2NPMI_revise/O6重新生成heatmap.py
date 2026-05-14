# -*- coding: utf-8 -*-
"""
生成核心意象排名演变热力图 (core_imageries_heatmap_Eng.png)
解决中文显示为方块的问题
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ==================== 用户配置区域 ====================
base_path = '按时期整理网络/整理的网络/'  # 四个时期文件夹的父目录
periods_original = ['初唐', '盛唐', '中唐', '晚唐']
periods_english = ['Early Tang', 'High Tang', 'Mid Tang', 'Late Tang']

# 翻译字典（请自行补充完整）
PHRASE_TRANSLATION = {
    '北斗': 'Big Dipper',
    '昆仑': 'Kunlun Mountains',
    '罗衣': 'silk robe',
    '宛转': 'lingering',
    '霓裳': 'rainbow gown',
    '翡翠': 'jadeite',
    '意气': 'lofty spirit',
    '杨柳': 'willow',
    '春风': 'spring breeze',
    '明月': 'bright moon',
    '江南': 'river south',
    '风光': 'scenery',
    '清风': 'clear wind',
    '化雨': 'nourishing rain',
    '何时': 'when',
    '相宜': 'suitable',
    '难忘': 'unforgettable',
    # 请根据实际 stable_phrases 补充
}

output_image = 'core_imageries_heatmap_Eng.png'


# ==================== 强制设置中文字体 ====================
def set_chinese_font():
    """尝试多种方法设置中文字体，确保 matplotlib 能显示中文"""
    # 方法1：直接指定字体名称（适用于 Windows 和常见 Linux 发行版）
    font_candidates = [
        'Microsoft YaHei',  # Windows
        'SimHei',  # Windows 黑体
        'WenQuanYi Micro Hei',  # Linux
        'Noto Sans CJK SC',  # Linux
        'PingFang SC',  # macOS
        'STHeiti',  # macOS
        'Arial Unicode MS'  # 备用
    ]

    for font in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            # 测试该字体是否可以显示中文
            fig, ax = plt.subplots(figsize=(0.1, 0.1))
            ax.text(0, 0, '测试中文', fontsize=8)
            plt.close(fig)
            print(f"✓ 成功加载中文字体: {font}")
            return
        except:
            continue

    # 如果都失败，尝试使用 font_manager 指定系统字体路径（Linux 常用）
    try:
        import matplotlib.font_manager as fm
        # 常见中文字体路径（可根据实际情况添加）
        font_paths = [
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            '/System/Library/Fonts/PingFang.ttc',
            'C:/Windows/Fonts/msyh.ttc',
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
                plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
                print(f"✓ 从文件加载字体: {fp}")
                return
    except:
        pass

    print("⚠️ 警告: 未能设置中文字体，中文将显示为方块。请手动安装中文字体。")


# 调用字体设置
set_chinese_font()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==================== 核心逻辑 ====================
def collect_top20_phrases():
    all_phrases = []
    for period in periods_original:
        node_path = os.path.join(base_path, period, 'node_centralities.csv')
        if not os.path.exists(node_path):
            print(f"警告: 文件不存在 {node_path}")
            continue
        nodes_df = pd.read_csv(node_path)
        top20 = nodes_df.nlargest(20, 'degree')['phrase'].tolist()
        all_phrases.extend(top20)
    return all_phrases


def get_rank_matrix(stable_phrases):
    rank_matrix = []
    for phrase in stable_phrases:
        row = []
        for period in periods_original:
            node_path = os.path.join(base_path, period, 'node_centralities.csv')
            if not os.path.exists(node_path):
                row.append(np.nan)
                continue
            nodes_df = pd.read_csv(node_path)
            top20 = nodes_df.nlargest(20, 'degree')['phrase'].tolist()
            if phrase in top20:
                rank = top20.index(phrase) + 1
                row.append(rank)
            else:
                row.append(np.nan)
        rank_matrix.append(row)
    return rank_matrix


def generate_heatmap():
    print("正在读取数据...")
    all_phrases = collect_top20_phrases()
    if not all_phrases:
        print("错误: 未找到任何 node_centralities.csv 文件，请检查路径配置。")
        return

    phrase_counts = Counter(all_phrases)
    stable_phrases = [p for p, c in phrase_counts.items() if c >= 2]
    stable_phrases = stable_phrases[:15]
    print(f"将显示以下 {len(stable_phrases)} 个核心意象:")
    print(stable_phrases)

    rank_matrix = get_rank_matrix(stable_phrases)

    # 生成带翻译的 Y 轴标签
    ytick_labels = []
    for p in stable_phrases:
        if p in PHRASE_TRANSLATION:
            label = f"{p} ({PHRASE_TRANSLATION[p]})"
        else:
            print(f"提示: 意象 '{p}' 未在翻译字典中，将只显示中文。")
            label = p
        ytick_labels.append(label)

    plt.figure(figsize=(10, 8))
    sns.heatmap(rank_matrix, annot=True, fmt='.0f', cmap='YlOrRd_r',
                xticklabels=periods_english, yticklabels=ytick_labels,
                cbar_kws={'label': 'Rank (smaller indicates higher importance)'})
    plt.title('Heatmap of Core Imagery Rank Evolution', fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('Imagery', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n✓ 热力图已保存至: {output_image}")
    plt.show()


if __name__ == "__main__":
    generate_heatmap()