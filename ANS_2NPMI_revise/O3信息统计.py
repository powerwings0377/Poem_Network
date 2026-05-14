import os
import pandas as pd
import re
from collections import Counter

# 设置路径
excel_path = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\年份\诗人年份-详细.xlsx"
source_dir = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\按诗人"

# 读取诗人时期对照表
print("正在读取诗人时期对照表...")
df = pd.read_excel(excel_path)
poet_period_dict = dict(zip(df['诗人名'], df['时期']))

# 初始化统计数据
stats = {
    'total': {
        'poems': 0,
        'poets': set(),
        'chars': 0,
        'vocab': set(),
        'char_counter': Counter()  # 添加字频统计
    },
    '初唐': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()},
    '盛唐': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()},
    '中唐': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()},
    '晚唐': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()},
    '未知': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()},
    '存疑': {'poems': 0, 'poets': set(), 'chars': 0, 'vocab': set(), 'char_counter': Counter()}
}


# 只保留中文字符的函数
def extract_chinese(text):
    # 只保留中文字符（Unicode范围：\u4e00-\u9fff）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return ''.join(chinese_chars)


# 遍历所有诗人文件夹
print("\n开始统计...")
total_poem_count = 0

for poet_folder in os.listdir(source_dir):
    poet_folder_path = os.path.join(source_dir, poet_folder)

    if not os.path.isdir(poet_folder_path):
        continue

    # 提取诗人名
    try:
        parts = poet_folder.split('-')
        poet_name = parts[1] if len(parts) >= 2 else None
    except:
        print(f"无法解析文件夹名: {poet_folder}")
        continue

    if not poet_name:
        continue

    # 获取诗人时期
    period = poet_period_dict.get(poet_name, '未知')
    if period not in stats:
        period = '未知'

    # 统计该诗人的所有诗
    poet_poem_count = 0
    poet_chars = 0
    poet_vocab = set()
    poet_counter = Counter()

    for poem_file in os.listdir(poet_folder_path):
        poem_file_path = os.path.join(poet_folder_path, poem_file)

        if not os.path.isfile(poem_file_path):
            continue

        try:
            # 读取诗内容
            with open(poem_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # 提取中文字符
            chinese_content = extract_chinese(content)
            char_count = len(chinese_content)

            # 获取该诗的所有不重复字符
            poem_chars = set(chinese_content)

            # 更新字频统计
            poet_counter.update(chinese_content)

            # 更新统计
            poet_poem_count += 1
            poet_chars += char_count
            poet_vocab.update(poem_chars)

        except Exception as e:
            print(f"读取文件出错 {poem_file_path}: {e}")
            continue

    # 更新时期统计
    stats[period]['poems'] += poet_poem_count
    stats[period]['poets'].add(poet_name)
    stats[period]['chars'] += poet_chars
    stats[period]['vocab'].update(poet_vocab)
    stats[period]['char_counter'].update(poet_counter)

    # 更新总体统计
    stats['total']['poems'] += poet_poem_count
    stats['total']['poets'].add(poet_name)
    stats['total']['chars'] += poet_chars
    stats['total']['vocab'].update(poet_vocab)
    stats['total']['char_counter'].update(poet_counter)

    total_poem_count += poet_poem_count
    print(f"已处理 {poet_name} ({period})：{poet_poem_count}首诗，{poet_chars}字，{len(poet_vocab)}个不重复字")

# 输出统计结果
print("\n" + "=" * 70)
print("古诗文语料统计报告（基于中文字符）")
print("=" * 70)

print(f"\n【总体统计】")
print(f"总诗篇数：{stats['total']['poems']:,} 篇")
print(f"总诗人数量：{len(stats['total']['poets'])} 位")
print(f"总字数（仅中文字符）：{stats['total']['chars']:,} 字")
print(f"总词汇表大小（不重复中文字符）：{len(stats['total']['vocab']):,} 个")

print(f"\n【分时期统计】")
print("-" * 70)
print(f"{'时期':<6} {'诗篇数':<12} {'诗人数量':<10} {'总字数':<15} {'词汇表大小':<12}")
print("-" * 70)

main_periods = ['初唐', '盛唐', '中唐', '晚唐']
for period in main_periods:
    p = stats[period]
    print(f"{period:<6} {p['poems']:<12,} {len(p['poets']):<10} {p['chars']:<15,} {len(p['vocab']):<12,}")

if stats['未知']['poems'] > 0:
    p = stats['未知']
    print(f"{'未知':<6} {p['poems']:<12,} {len(p['poets']):<10} {p['chars']:<15,} {len(p['vocab']):<12,}")
if stats['存疑']['poems'] > 0:
    p = stats['存疑']
    print(f"{'存疑':<6} {p['poems']:<12,} {len(p['poets']):<10} {p['chars']:<15,} {len(p['vocab']):<12,}")

print("-" * 70)

# 计算百分比
total_main_poems = sum(stats[p]['poems'] for p in main_periods)
total_main_poets = sum(len(stats[p]['poets']) for p in main_periods)
total_main_chars = sum(stats[p]['chars'] for p in main_periods)
total_main_vocab = len(set.union(*[stats[p]['vocab'] for p in main_periods]))

print(f"\n【各时期占比】（基于主要时期合计）")
print("-" * 70)
print(f"{'时期':<6} {'诗篇占比':<15} {'诗人占比':<15} {'字数占比':<15} {'词汇占比':<15}")
print("-" * 70)

for period in main_periods:
    p = stats[period]
    poem_pct = p['poems'] / total_main_poems * 100
    poet_pct = len(p['poets']) / total_main_poets * 100
    char_pct = p['chars'] / total_main_chars * 100
    vocab_pct = len(p['vocab']) / total_main_vocab * 100

    print(
        f"{period:<6} {poem_pct:>8.2f}% ({p['poems']:<8,})  {poet_pct:>8.2f}% ({len(p['poets']):<4})  {char_pct:>8.2f}% ({p['chars']:<10,})  {vocab_pct:>8.2f}% ({len(p['vocab']):<6,})")

print("-" * 70)

# 平均指标
print(f"\n【平均指标】")
print(f"平均每首诗字数：{stats['total']['chars'] / stats['total']['poems']:.1f} 字")
print(f"平均每位诗人作品数：{stats['total']['poems'] / len(stats['total']['poets']):.1f} 篇")
print(f"平均每位诗人贡献字数：{stats['total']['chars'] / len(stats['total']['poets']):,.0f} 字")

print(f"\n【分时期平均指标】")
print("-" * 50)
print(f"{'时期':<6} {'平均诗长(字)':<15} {'人均诗篇数':<12} {'人均用字量':<12}")
print("-" * 50)

for period in main_periods:
    p = stats[period]
    if p['poems'] > 0 and len(p['poets']) > 0:
        avg_poem_len = p['chars'] / p['poems']
        avg_poems_per_poet = p['poems'] / len(p['poets'])
        avg_chars_per_poet = p['chars'] / len(p['poets'])
        print(f"{period:<6} {avg_poem_len:>12.1f}      {avg_poems_per_poet:>8.1f}      {avg_chars_per_poet:>10,.0f}")

print("-" * 50)

# 字频统计
print(f"\n【总体最常用20个汉字】")
total_top20 = stats['total']['char_counter'].most_common(20)
print(f"{'排名':<4} {'汉字':<6} {'出现次数':<12} {'占比':<10}")
print("-" * 40)
for i, (char, count) in enumerate(total_top20, 1):
    percentage = count / stats['total']['chars'] * 100
    print(f"{i:<4} {char:<6} {count:<12,} {percentage:.4f}%")

print(f"\n【分时期最常用10个汉字对比】")
print("-" * 70)
for period in main_periods:
    p = stats[period]
    if p['chars'] > 0:
        print(f"\n{period}时期最常用10个汉字：")
        top10 = p['char_counter'].most_common(10)
        print(f"{'排名':<4} {'汉字':<6} {'出现次数':<12} {'时期占比':<10} {'总体占比':<10}")
        print("-" * 50)
        for i, (char, count) in enumerate(top10, 1):
            period_pct = count / p['chars'] * 100
            total_pct = stats['total']['char_counter'][char] / stats['total']['chars'] * 100
            print(f"{i:<4} {char:<6} {count:<12,} {period_pct:.4f}%    {total_pct:.4f}%")

# 将统计结果保存到文件
output_file = os.path.join(os.path.dirname(source_dir), "语料统计报告_字符版.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("古诗文语料统计报告（基于中文字符）\n")
    f.write("=" * 60 + "\n\n")

    f.write("【总体统计】\n")
    f.write(f"总诗篇数：{stats['total']['poems']:,} 篇\n")
    f.write(f"总诗人数量：{len(stats['total']['poets'])} 位\n")
    f.write(f"总字数（仅中文字符）：{stats['total']['chars']:,} 字\n")
    f.write(f"总词汇表大小（不重复中文字符）：{len(stats['total']['vocab']):,} 个\n\n")

    f.write("【分时期统计】\n")
    f.write(f"{'时期':<6} {'诗篇数':<10} {'诗人数量':<8} {'总字数':<12} {'词汇表大小':<10}\n")
    f.write("-" * 50 + "\n")
    for period in main_periods:
        p = stats[period]
        f.write(f"{period:<6} {p['poems']:<10,} {len(p['poets']):<8} {p['chars']:<12,} {len(p['vocab']):<10,}\n")

    f.write("\n【总体最常用20个汉字】\n")
    for i, (char, count) in enumerate(total_top20, 1):
        percentage = count / stats['total']['chars'] * 100
        f.write(f"{i:2d}. {char} : {count:,}次 ({percentage:.4f}%)\n")

    f.write("\n【分时期最常用10个汉字】\n")
    for period in main_periods:
        p = stats[period]
        if p['chars'] > 0:
            f.write(f"\n{period}时期：\n")
            top10 = p['char_counter'].most_common(10)
            for i, (char, count) in enumerate(top10, 1):
                period_pct = count / p['chars'] * 100
                f.write(f"  {i:2d}. {char} : {count:,}次 ({period_pct:.4f}%)\n")

print(f"\n统计报告已保存至：{output_file}")
print("统计完成！")