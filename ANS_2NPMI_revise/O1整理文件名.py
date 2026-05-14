import os
import shutil
from pathlib import Path


def preprocess_filename(filename):
    """
    预处理文件名：给以"-"结尾的文件添加"未知"作为诗人
    然后进行分割处理
    """
    # 如果文件名以"-"结尾，则添加"未知"
    if filename.endswith('-'):
        processed_name = filename + '未知'
    else:
        processed_name = filename

    # 分割文件名
    parts = processed_name.split("-")

    # 获取最后一个非空部分作为诗人名
    poet_name = ""
    for part in reversed(parts):
        if part.strip():
            poet_name = part.strip()
            break

    return poet_name, processed_name


def organize_poems_by_poet_with_numbering():
    """
    将唐诗文件按诗人分类整理，文件夹名包含序号和篇数统计

    处理规则:
    1. 先预处理文件名：给以"-"结尾的文件添加"未知"作为诗人
    2. 文件名按"-"分割，最后一个非空部分作为诗人名
    3. 按诗人作品数量从多到少排序，分配序号
    4. 创建格式为"0001-诗人名-篇数篇"的文件夹
    5. 将文件复制到对应的诗人文件夹中
    """

    # 源文件夹路径
    source_dir = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\Tang_Poems"

    # 目标文件夹路径
    target_base_dir = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\按诗人"

    # 创建目标文件夹（如果不存在）
    Path(target_base_dir).mkdir(parents=True, exist_ok=True)

    # 第一阶段：统计每个诗人的文件数量
    print("第一阶段：统计诗人作品数量...")
    print("=" * 60)

    poet_file_counts = {}
    poet_original_names = {}  # 记录诗人原名（如果有变化的话）
    total_files = 0
    preprocessed_count = 0

    try:
        # 获取所有文件
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        total_files = len(files)

        if total_files == 0:
            print("错误: 源文件夹中没有找到文件！")
            return

        print(f"找到 {total_files} 个文件")

        # 显示预处理示例
        print("\n文件名预处理示例:")
        print("-" * 50)

        # 先找几个示例文件
        example_files = []
        for filename in files[:10]:  # 检查前10个文件
            if filename.endswith('-'):
                example_files.append(filename)
                if len(example_files) >= 3:
                    break

        for example in example_files:
            poet_name, processed_name = preprocess_filename(example)
            print(f"原始文件名: {example}")
            print(f"处理后诗人名: {poet_name}")
            print(f"处理后的名称: {processed_name}")
            print()

        # 统计每个诗人的文件数量
        for filename in files:
            # 预处理文件名
            poet_name, processed_name = preprocess_filename(filename)

            # 记录原始名称（如果发生变化）
            original_poet = filename.split("-")[-1] if filename.split("-")[-1].strip() else "空"
            if original_poet != poet_name:
                poet_original_names[poet_name] = poet_original_names.get(poet_name, set())
                poet_original_names[poet_name].add(original_poet)

            # 更新统计
            poet_file_counts[poet_name] = poet_file_counts.get(poet_name, 0) + 1

            # 统计预处理的数量
            if filename.endswith('-'):
                preprocessed_count += 1

        # 按文件数量从多到少排序
        sorted_poets = sorted(poet_file_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"共发现 {len(sorted_poets)} 位诗人")
        print(f"需要预处理的文件数: {preprocessed_count} 个（以'-'结尾的文件）")

        # 显示原始名称变化情况
        if poet_original_names:
            print("\n诗人名称预处理情况:")
            print("-" * 50)
            for poet, original_set in sorted(poet_original_names.items()):
                if original_set:
                    originals = list(original_set)
                    if len(originals) > 3:
                        display = f"{', '.join(originals[:3])}...（共{len(originals)}种）"
                    else:
                        display = ', '.join(originals)
                    print(f"'{poet}' <- 来自: {display}")

        print("\n诗人作品数量排名（前20位）:")
        print("-" * 50)

        for i, (poet, count) in enumerate(sorted_poets[:20], 1):
            print(f"{i:3d}. {poet:<10} : {count:>4} 篇")

        if len(sorted_poets) > 20:
            print(f"... 还有 {len(sorted_poets) - 20} 位诗人")

    except FileNotFoundError:
        print(f"错误: 找不到文件夹 '{source_dir}'，请检查路径是否正确")
        return
    except PermissionError:
        print("错误: 没有文件访问权限，请以管理员权限运行")
        return
    except Exception as e:
        print(f"错误: 统计过程中发生错误: {e}")
        return

    # 第二阶段：创建带有序号和篇数统计的文件夹并复制文件
    print("\n" + "=" * 60)
    print("第二阶段：创建文件夹并复制文件...")
    print("=" * 60)

    # 创建诗人名称到文件夹名的映射
    poet_to_folder = {}
    folder_info = []  # 存储文件夹信息

    for i, (poet, count) in enumerate(sorted_poets, 1):
        # 格式化序号（4位数字，不足补0）
        seq_num = f"{i:04d}"
        # 创建文件夹名：序号-诗人名-篇数篇
        folder_name = f"{seq_num}-{poet}-{count}篇"
        poet_to_folder[poet] = folder_name
        folder_info.append((seq_num, poet, count, folder_name))

    # 显示将要创建的文件夹信息
    print("将要创建的文件夹（前30个）:")
    print("-" * 50)
    for seq_num, poet, count, folder_name in folder_info[:30]:  # 显示前30个
        print(f"{folder_name}")

    if len(folder_info) > 30:
        print(f"... 还有 {len(folder_info) - 30} 个文件夹")

    # 确认是否继续
    confirm = input("\n是否继续创建文件夹并复制文件？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return

    # 开始创建文件夹和复制文件
    processed_files = 0

    try:
        for filename in files:
            source_path = os.path.join(source_dir, filename)

            # 预处理文件名获取诗人信息
            poet_name, _ = preprocess_filename(filename)

            # 获取对应的文件夹名
            folder_name = poet_to_folder.get(poet_name)
            if not folder_name:
                print(f"警告: 未找到诗人 '{poet_name}' 的文件夹映射")
                continue

            # 创建诗人文件夹
            poet_dir = os.path.join(target_base_dir, folder_name)
            Path(poet_dir).mkdir(parents=True, exist_ok=True)

            # 目标文件路径
            target_path = os.path.join(poet_dir, filename)

            # 复制文件
            shutil.copy2(source_path, target_path)
            processed_files += 1

            # 显示进度（每处理50个文件显示一次）
            if processed_files % 50 == 0:
                print(f"已处理 {processed_files}/{total_files} 个文件...")

    except Exception as e:
        print(f"错误: 复制文件过程中发生错误: {e}")
        return

    # 显示最终统计结果
    print("\n" + "=" * 60)
    print("文件整理完成！")
    print("=" * 60)

    # 创建详细统计报告
    report_file = os.path.join(target_base_dir, "诗人统计报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("唐诗诗人作品统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总文件数: {total_files} 篇\n")
        f.write(f"诗人总数: {len(sorted_poets)} 位\n")
        f.write(f"预处理的文件数: {preprocessed_count} 个（以'-'结尾的文件）\n\n")

        # 写入预处理规则说明
        f.write("预处理规则说明:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 对于以'-'结尾的文件名，自动添加'未知'作为诗人名\n")
        f.write("2. 例如：'00545-第一十三卷-卷.25-郊庙歌辞·武后享清庙乐章十首·第六送文舞-'\n")
        f.write("   会被处理为诗人名='未知'\n")
        f.write("3. 其他文件正常提取最后一个非空部分作为诗人名\n\n")

        # 写入名称变化情况
        if poet_original_names:
            f.write("诗人名称预处理情况:\n")
            f.write("-" * 40 + "\n")
            for poet, original_set in sorted(poet_original_names.items()):
                if original_set:
                    originals = sorted(list(original_set))
                    f.write(f"'{poet}' <- 来自: {', '.join(originals)}\n")
            f.write("\n")

        f.write("诗人作品数量排名:\n")
        f.write("-" * 60 + "\n")
        f.write("序号 | 诗人名 | 作品数量 | 文件夹名\n")
        f.write("-" * 60 + "\n")

        for i, (poet, count) in enumerate(sorted_poets, 1):
            seq_num = f"{i:04d}"
            folder_name = f"{seq_num}-{poet}-{count}篇"
            f.write(f"{seq_num:>4} | {poet:<10} | {count:>8} | {folder_name}\n")

        # 写入一些统计信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计摘要:\n")
        f.write("-" * 40 + "\n")
        f.write(f"作品最多的诗人: {sorted_poets[0][0]} ({sorted_poets[0][1]}篇)\n")

        if len(sorted_poets) > 1:
            f.write(f"作品第二多的诗人: {sorted_poets[1][0]} ({sorted_poets[1][1]}篇)\n")

        # 统计各数量区间的诗人数量
        ranges = {
            "1000篇以上": 0,
            "500-999篇": 0,
            "100-499篇": 0,
            "50-99篇": 0,
            "10-49篇": 0,
            "1-9篇": 0
        }

        for poet, count in sorted_poets:
            if count >= 1000:
                ranges["1000篇以上"] += 1
            elif count >= 500:
                ranges["500-999篇"] += 1
            elif count >= 100:
                ranges["100-499篇"] += 1
            elif count >= 50:
                ranges["50-99篇"] += 1
            elif count >= 10:
                ranges["10-49篇"] += 1
            else:
                ranges["1-9篇"] += 1

        f.write("\n诗人作品数量分布:\n")
        for range_name, count in ranges.items():
            if count > 0:
                f.write(f"  {range_name}: {count} 位诗人\n")

    # 显示最终结果
    print(f"总文件数: {total_files} 篇")
    print(f"成功处理: {processed_files} 篇")
    print(f"需要预处理的文件数: {preprocessed_count} 篇")
    print(f"创建的诗人文件夹数: {len(sorted_poets)} 个")

    # 显示预处理规则
    print(f"\n预处理规则: 以'-'结尾的文件名会自动添加'未知'作为诗人名")
    print(f"例如: '00545-...-第六送文舞-' -> 诗人名='未知'")

    # 显示文件夹命名格式
    print(f"\n文件夹命名格式: 序号(4位)-诗人名-篇数篇")
    print(f"例如: 0001-李白-120篇")

    # 显示前10位诗人的文件夹
    print("\n前10位诗人的文件夹:")
    print("-" * 50)
    for i, (seq_num, poet, count, folder_name) in enumerate(folder_info[:10], 1):
        print(f"{folder_name}")

    # 显示未知诗人的统计（如果有）
    unknown_count = poet_file_counts.get("未知", 0)
    if unknown_count > 0:
        unknown_index = next((i for i, (poet, _) in enumerate(sorted_poets, 1) if poet == "未知"), None)
        if unknown_index:
            print(f"\n未知诗人: 共 {unknown_count} 篇，排在 {unknown_index:04d} 位")

    print(f"\n详细统计报告已保存到: {report_file}")


def preview_with_numbering():
    """
    预览带序号的文件整理情况
    """
    source_dir = r"F:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\Tang_Poems"

    print("预览带序号的文件夹整理方案（含预处理）:")
    print("=" * 70)

    try:
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        if not files:
            print("错误: 源文件夹中没有找到文件！")
            return

        poet_file_counts = {}
        preprocessed_examples = []

        # 统计每个诗人的文件数量
        for filename in files:
            # 预处理文件名
            poet_name, processed_name = preprocess_filename(filename)

            # 收集预处理示例
            if filename.endswith('-') and len(preprocessed_examples) < 5:
                preprocessed_examples.append((filename, poet_name, processed_name))

            poet_file_counts[poet_name] = poet_file_counts.get(poet_name, 0) + 1

        # 按文件数量排序
        sorted_poets = sorted(poet_file_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"总文件数: {len(files)} 篇")
        print(f"诗人总数: {len(sorted_poets)} 位")

        # 显示预处理示例
        if preprocessed_examples:
            print("\n文件名预处理示例:")
            print("-" * 70)
            for original, poet_name, processed in preprocessed_examples:
                print(f"原始文件名: {original}")
                print(f"处理后的诗人名: {poet_name}")
                print(f"处理后的完整名称: {processed}")
                print()

        # 显示前20位诗人的文件夹命名方案
        print("\n前20位诗人的文件夹命名方案:")
        print("-" * 70)
        print("序号 | 诗人名 | 作品数量 | 文件夹名")
        print("-" * 70)

        for i, (poet, count) in enumerate(sorted_poets[:20], 1):
            seq_num = f"{i:04d}"
            folder_name = f"{seq_num}-{poet}-{count}篇"
            print(f"{seq_num:>4} | {poet:<10} | {count:>8} | {folder_name}")

        # 显示未知诗人的信息
        unknown_count = poet_file_counts.get("未知", 0)
        if unknown_count > 0:
            unknown_index = next((i for i, (poet, _) in enumerate(sorted_poets, 1) if poet == "未知"), None)
            print(f"\n未知诗人: 共 {unknown_count} 篇，将排在 {unknown_index:04d} 位")
            print(f"文件夹名: {unknown_index:04d}-未知-{unknown_count}篇")

        print("\n这只是预览，要实际整理文件请运行主程序")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("唐诗文件整理工具（带预处理和序号统计）")
    print("=" * 70)
    print("功能特点:")
    print("1. 预处理: 给以'-'结尾的文件名添加'未知'作为诗人名")
    print("2. 智能排序: 按诗人作品数量从多到少排序")
    print("3. 文件夹命名: 序号(4位)-诗人名-篇数篇")
    print("=" * 70)
    print("1. 实际整理文件（创建带序号的文件夹）")
    print("2. 预览整理方案（不复制文件）")
    print("3. 退出")
    print("=" * 70)

    choice = input("请选择 (1/2/3): ").strip()

    if choice == "1":
        organize_poems_by_poet_with_numbering()
    elif choice == "2":
        preview_with_numbering()
    elif choice == "3":
        print("退出程序")
    else:
        print("无效选择，退出程序")