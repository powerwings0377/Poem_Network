import os
import pandas as pd
import shutil
from pathlib import Path

# 设置路径
excel_path = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\年份\诗人年份-详细.xlsx"
source_dir = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\按诗人"
target_base = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\按时期"

# 读取Excel文件
print("正在读取诗人时期对照表...")
df = pd.read_excel(excel_path)

# 确保有诗人名和时期两列（根据你的实际列名调整）
# 假设列名为"诗人名"和"时期"，如果不同请修改
poet_period_dict = dict(zip(df['诗人名'], df['时期']))

# 创建四个时期文件夹
periods = ['初唐', '盛唐', '中唐', '晚唐']
for period in periods:
    period_path = os.path.join(target_base, period)
    os.makedirs(period_path, exist_ok=True)
    print(f"创建文件夹: {period_path}")

# 遍历源文件夹中的所有诗人文件夹
print("\n开始复制文件...")
copied_count = 0
error_count = 0

for poet_folder in os.listdir(source_dir):
    poet_folder_path = os.path.join(source_dir, poet_folder)

    # 只处理文件夹
    if not os.path.isdir(poet_folder_path):
        continue

    # 从文件夹名中提取诗人名（格式如"0001-白居易-2627篇"）
    try:
        # 按"-"分割，第二部分是诗人名
        parts = poet_folder.split('-')
        if len(parts) >= 2:
            poet_name = parts[1]
        else:
            print(f"无法解析诗人文件夹名: {poet_folder}")
            continue
    except Exception as e:
        print(f"解析文件夹名出错 {poet_folder}: {e}")
        continue

    # 查找诗人对应的时期
    period = poet_period_dict.get(poet_name)

    if period not in periods:
        print(f"诗人 {poet_name} 的时期 '{period}' 不在预期范围内，跳过")
        error_count += 1
        continue

    # 目标时期文件夹路径
    target_period_path = os.path.join(target_base, period)

    # 遍历该诗人文件夹下的所有诗文件
    for poem_file in os.listdir(poet_folder_path):
        poem_file_path = os.path.join(poet_folder_path, poem_file)

        # 只处理文件（不处理子文件夹）
        if not os.path.isfile(poem_file_path):
            continue

        # 复制文件到对应的时期文件夹
        target_file_path = os.path.join(target_period_path, poem_file)

        # 处理文件名冲突（如果不同诗人有同名文件）
        if os.path.exists(target_file_path):
            # 在文件名前加上诗人名前缀
            name, ext = os.path.splitext(poem_file)
            new_filename = f"{poet_name}-{name}{ext}"
            target_file_path = os.path.join(target_period_path, new_filename)

        try:
            shutil.copy2(poem_file_path, target_file_path)
            copied_count += 1
            if copied_count % 100 == 0:  # 每100个文件打印一次进度
                print(f"已复制 {copied_count} 个文件...")
        except Exception as e:
            print(f"复制文件出错 {poem_file_path}: {e}")
            error_count += 1

print(f"\n任务完成！")
print(f"成功复制文件数: {copied_count}")
print(f"错误/跳过数: {error_count}")
print(f"\n文件已复制到: {target_base}")
print("各时期文件夹: 初唐、盛唐、中唐、晚唐")