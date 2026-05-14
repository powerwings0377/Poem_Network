import numpy as np
import matplotlib.pyplot as plt


# ========== 1. 读取数据 ==========
def read_data(file_path):
    """读取数据文件"""
    words = []
    npmis = []
    freqs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if lines and ('排名' in lines[0] or '词语' in lines[0]):
        lines = lines[1:]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '\t' in line:
            parts = line.split('\t')
        else:
            parts = line.split()

        if len(parts) >= 4:
            try:
                word = parts[1].strip()
                npmi = float(parts[2])
                freq = int(float(parts[3]))

                words.append(word)
                npmis.append(npmi)
                freqs.append(freq)

            except (ValueError, IndexError):
                continue

    return words, npmis, freqs


# ========== 2. 绘制传统直方图 ==========
def plot_traditional_histogram(npmis, freqs, threshold=0.3):
    """绘制传统样式的直方图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 定义NPMI区间
    bins = np.arange(-0.2, 1.05, 0.1)  # 从-0.2到1.0，每0.1一个区间

    # ===== 图1：词语数量直方图 =====
    # 使用matplotlib的hist函数绘制传统直方图
    n1, bins1, patches1 = ax1.hist(npmis, bins=bins,
                                   edgecolor='black', alpha=0.7)

    # 设置柱子颜色
    for i, patch in enumerate(patches1):
        bin_center = (bins1[i] + bins1[i + 1]) / 2
        if bin_center < threshold:
            patch.set_facecolor('lightcoral')
        else:
            patch.set_facecolor('lightgreen')

    # 添加阈值线
    ax1.axvline(x=threshold, color='red', linestyle='--',
                linewidth=2, label=f'阈值 θ={threshold}')

    # 添加数值标签
    for i in range(len(n1)):
        if n1[i] > 0:
            # 在柱子中心上方添加标签
            bin_center = (bins1[i] + bins1[i + 1]) / 2
            ax1.text(bin_center, n1[i] + max(n1) * 0.02,
                     f'{int(n1[i]):,}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # 设置图1的坐标轴
    ax1.set_xlabel('NPMI值', fontsize=11)
    ax1.set_ylabel('词语数量', fontsize=11)
    ax1.set_title('NPMI值分布直方图（按词语数量）', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_xlim(-0.25, 1.05)
    ax1.legend(loc='upper right')

    # 设置x轴刻度为区间边界值
    ax1.set_xticks(bins)

    # ===== 图2：频率总和直方图 =====
    # 使用hist函数，但用weights参数考虑频率
    n2, bins2, patches2 = ax2.hist(npmis, bins=bins, weights=freqs,
                                   edgecolor='black', alpha=0.7)

    # 设置柱子颜色
    for i, patch in enumerate(patches2):
        bin_center = (bins2[i] + bins2[i + 1]) / 2
        if bin_center < threshold:
            patch.set_facecolor('lightcoral')
        else:
            patch.set_facecolor('lightblue')

    # 添加阈值线
    ax2.axvline(x=threshold, color='red', linestyle='--',
                linewidth=2, label=f'阈值 θ={threshold}')

    # 添加数值标签
    for i in range(len(n2)):
        if n2[i] > 0:
            bin_center = (bins2[i] + bins2[i + 1]) / 2
            ax2.text(bin_center, n2[i] + max(n2) * 0.02,
                     f'{int(n2[i]):,}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # 设置图2的坐标轴
    ax2.set_xlabel('NPMI值', fontsize=11)
    ax2.set_ylabel('频率总和', fontsize=11)
    ax2.set_title('NPMI值分布直方图（按频率总和）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xlim(-0.25, 1.05)
    ax2.set_xticks(bins)
    ax2.legend(loc='upper right')

    # 格式化y轴标签（添加千位分隔符）
    ax2.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # 调整布局
    plt.tight_layout()

    # 计算并显示统计数据
    print("\n" + "=" * 60)
    print("直方图统计数据:")
    print("=" * 60)
    print(f"{'NPMI区间':<20} {'词语数量':<15} {'频率总和':<15}")
    print("-" * 60)

    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]

        # 计算词语数量
        word_count = sum(1 for npmi in npmis if bin_min <= npmi < bin_max)
        # 计算频率总和
        freq_sum = sum(freq for npmi, freq in zip(npmis, freqs)
                       if bin_min <= npmi < bin_max)

        if word_count > 0 or freq_sum > 0:
            print(f"[{bin_min:.1f},{bin_max:.1f})".ljust(20) +
                  f"{word_count:,}".ljust(15) +
                  f"{freq_sum:,}")

    return fig, (ax1, ax2)


# ========== 3. 主程序 ==========
def main():
    # 读取数据文件
    file_path = 'npmi_results_all_2gram.txt'

    print(f"正在读取文件: {file_path}")
    print("=" * 60)

    try:
        # 读取数据
        words, npmis, freqs = read_data(file_path)

        if len(words) == 0:
            print("错误：未能读取到任何数据")
            return

        # 基本统计
        print(f"\n数据读取成功！")
        print(f"  总词语数量: {len(words):,}")
        print(f"  总频率: {sum(freqs):,}")
        print(f"  NPMI最小值: {min(npmis):.6f}")
        print(f"  NPMI最大值: {max(npmis):.6f}")
        print(f"  NPMI平均值: {np.mean(npmis):.6f}")
        print(f"  NPMI中位数: {np.median(npmis):.6f}")

        # 绘制传统直方图
        fig, axes = plot_traditional_histogram(npmis, freqs)

        # 保存图形
        output_file = 'NPMI_Traditional_Histogram.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图形已保存为 '{output_file}'")

        # 显示图形
        plt.show()

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 4. 运行程序 ==========
if __name__ == "__main__":
    main()