#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPMI计算程序 - Python版本
计算诗歌文档集中双字词的标准化互信息
输入：诗歌文件夹 + 词表文件
输出：NPMI结果CSV、词频统计CSV、详细报告TXT
"""

import os
import sys
import glob
import math
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path


# ==================== 工具类 ====================

class FileUtils:
    """文件工具类"""

    @staticmethod
    def read_text_file(file_path):
        """读取文本文件（UTF-8编码）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def read_word_list(file_path):
        """读取词表文件（每行一个词）"""
        all_lines = FileUtils.read_text_file(file_path).splitlines()
        word_list = []

        for line in all_lines:
            word = line.strip()
            if not word:  # 跳过空行
                continue
            if len(word) != 2:  # 检查是否为双字词
                print(f"警告：跳过非双字词 '{word}' (长度: {len(word)})")
                continue
            if word not in word_list:  # 去重
                word_list.append(word)

        return word_list

    @staticmethod
    def read_poems_from_folder(folder_path):
        """从文件夹读取所有诗歌文件（读取所有文件）"""
        poems = []
        poem_files = []

        if not os.path.isdir(folder_path):
            print(f"错误：文件夹不存在 {folder_path}")
            return poems, poem_files

        # 获取所有文件（包括无扩展名的）
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # 只处理文件，不处理子文件夹
            if os.path.isfile(file_path):
                try:
                    # 尝试以UTF-8读取
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    poems.append(content)
                    poem_files.append(filename)
                    print(f"已读取: {filename}")
                except Exception as e:
                    print(f"警告：读取文件 {file_path} 失败: {e}")

        print(f"共读取 {len(poems)} 个文件")
        return poems, poem_files

    @staticmethod
    def save_to_file(content, file_path):
        """保存字符串到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


class TextPreprocessor:
    """文本预处理工具"""

    @staticmethod
    def remove_line_breaks_only(text):
        """只移除换行符，保留所有其他字符"""
        if not text:
            return ""
        return text.replace('\r\n', '').replace('\n', '').replace('\r', '')

    @staticmethod
    def keep_chinese_only(text):
        """只保留汉字"""
        if not text:
            return ""
        return ''.join(c for c in text if '\u4e00' <= c <= '\u9fff')

    @staticmethod
    def print_preprocess_info(original_text, cleaned_text, original_files=None):
        """打印预处理信息"""
        print("=== 文本预处理信息 ===")
        if original_files:
            print(f"诗歌文件数量: {len(original_files)}")
        print(f"原始文本总长度: {len(original_text)} 字符")
        print(f"处理后总长度: {len(cleaned_text)} 字符")

        # 统计汉字数量
        chinese_count = sum(1 for c in original_text if '\u4e00' <= c <= '\u9fff')
        print(f"汉字数量: {chinese_count}")
        print(f"非汉字数量: {len(original_text) - chinese_count}")

        # 统计换行符数量
        line_break_count = original_text.count('\n') + original_text.count('\r')
        print(f"换行符数量: {line_break_count}")

        print(f"移除字符数量: {len(original_text) - len(cleaned_text)}")

        # 显示前后对比（前50字符）
        print("\n【原始文本前50字符】:")
        preview_original = original_text[:50] + "..." if len(original_text) > 50 else original_text
        print(f"  {preview_original.replace(chr(10), '\\n').replace(chr(13), '\\r')}")

        print("【处理后文本前50字符】:")
        preview_cleaned = cleaned_text[:50] + "..." if len(cleaned_text) > 50 else cleaned_text
        print(f"  {preview_cleaned}")
        print()


class Pair:
    """无序词对，用于作为字典的键"""

    def __init__(self, first, second):
        # 保证无序性：按字典序排序
        if first < second:
            self.first = first
            self.second = second
        else:
            self.first = second
            self.second = first

    def __eq__(self, other):
        if not isinstance(other, Pair):
            return False
        return self.first == other.first and self.second == other.second

    def __hash__(self):
        return hash((self.first, self.second))

    def __str__(self):
        return f"{self.first},{self.second}"

    def __repr__(self):
        return str(self)


# ==================== NPMI计算器 ====================

class PMICalculator:
    """基于文档的NPMI计算器"""

    def __init__(self, poems, word_list, poem_names=None):
        """
        初始化计算器

        Args:
            poems: 诗歌文本列表，每首诗是一个字符串
            word_list: 词表列表
            poem_names: 诗歌文件名列表（可选）
        """
        self.poems = poems
        self.word_list = word_list
        self.word_set = set(word_list)
        self.poem_names = poem_names if poem_names else [f"诗{i + 1}" for i in range(len(poems))]

        # 文档级统计（关键修改：每首诗是一个文档）
        self.doc_count = len(poems)  # 总文档数（总诗歌数）

        # 词在文档中的出现统计（一个文档中出现多次只计一次）
        self.word_doc_count = defaultdict(int)  # 包含该词的文档数
        self.pair_doc_count = defaultdict(int)  # 同时包含两个词的文档数

        # 用于输出兼容的统计（保持与原格式一致）
        self.word_total_occurrences = defaultdict(int)  # 词在文本中的总出现次数（用于输出）
        self.total_matches = 0  # 总匹配次数

        # NPMI结果
        self.npmi_results = {}  # Pair -> NPMI值
        self.calculation_time = 0

        # 预处理所有诗歌
        self.cleaned_poems = []
        for poem in poems:
            cleaned = TextPreprocessor.remove_line_breaks_only(poem)
            self.cleaned_poems.append(cleaned)

        # 合并所有文本（用于统计总字符数等）
        self.combined_text = ''.join(self.cleaned_poems)

    def calculate(self):
        """执行NPMI计算"""
        start_time = time.time()

        # 1. 统计文档级出现
        self._count_document_frequencies()

        # 2. 计算NPMI
        self._calculate_npmi()

        self.calculation_time = time.time() - start_time

    def _count_document_frequencies(self):
        """
        统计文档级出现（关键修改）
        每首诗作为一个文档，统计：
        - 包含某词的文档数
        - 同时包含两个词的文档数
        - 词的总出现次数（用于输出）
        """
        # 重置计数器
        self.word_doc_count.clear()
        self.pair_doc_count.clear()
        self.word_total_occurrences.clear()

        # 遍历每首诗
        for poem_idx, poem in enumerate(self.cleaned_poems):
            # 找出当前诗中出现的所有词（去重，用于文档级统计）
            words_in_this_poem = set()

            # 扫描整首诗，找出所有匹配的词
            for i in range(len(poem) - 1):
                two_char = poem[i:i + 2]
                if two_char in self.word_set:
                    words_in_this_poem.add(two_char)
                    # 同时统计总出现次数（用于输出）
                    self.word_total_occurrences[two_char] += 1
                    self.total_matches += 1

            # 文档级统计：每个词在文档中出现，文档计数+1
            for word in words_in_this_poem:
                self.word_doc_count[word] += 1

            # 统计词对共现（在同一首诗中）
            word_list_in_poem = list(words_in_this_poem)
            for i in range(len(word_list_in_poem)):
                for j in range(i + 1, len(word_list_in_poem)):
                    pair = Pair(word_list_in_poem[i], word_list_in_poem[j])
                    self.pair_doc_count[pair] += 1

    def _calculate_npmi(self):
        """计算NPMI值（使用文档级概率）"""
        self.npmi_results.clear()
        N = self.doc_count  # 总文档数

        if N == 0:
            print("警告：总文档数为0，无法计算NPMI")
            return

        for pair, co_count in self.pair_doc_count.items():
            word1 = pair.first
            word2 = pair.second

            count1 = self.word_doc_count.get(word1, 0)
            count2 = self.word_doc_count.get(word2, 0)

            if count1 == 0 or count2 == 0:
                continue

            # 文档级概率
            P1 = count1 / N
            P2 = count2 / N
            P12 = co_count / N

            if P1 * P2 == 0 or P12 == 0:
                continue

            # 计算PMI
            pmi = math.log(P12 / (P1 * P2))

            # 计算NPMI：PMI / (-log(P(x,y)))
            npmi = pmi / (-math.log(P12))

            # 确保NPMI在[-1, 1]范围内
            npmi = max(-1.0, min(1.0, npmi))

            if math.isfinite(npmi):
                self.npmi_results[pair] = npmi

    def get_sorted_npmi_results(self):
        """获取按NPMI值排序的结果（从高到低）"""
        sorted_items = sorted(self.npmi_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def get_missing_words(self):
        """获取未在文本中出现的词"""
        missing = []
        for word in self.word_list:
            if word not in self.word_total_occurrences or self.word_total_occurrences[word] == 0:
                missing.append(word)
        return missing

    def get_top_words(self, top_n):
        """获取出现次数最多的词"""
        sorted_words = sorted(self.word_total_occurrences.items(),
                              key=lambda x: x[1], reverse=True)
        return sorted_words[:min(top_n, len(sorted_words))]

    def print_statistics(self):
        """打印统计信息"""
        print("=== NPMI计算统计信息 ===")
        print(f"诗歌数量: {self.doc_count}")
        print(f"文本总长度: {len(self.combined_text)} 汉字")
        print(f"词表大小: {len(self.word_list)} 个词")
        print(f"总匹配次数: {self.total_matches}")
        print(f"有共现的词对数: {len(self.npmi_results)}")
        print(f"计算耗时: {self.calculation_time * 1000:.2f} ms")

        # 检查未出现的词
        missing_words = self.get_missing_words()
        if missing_words:
            print(f"警告: {len(missing_words)} 个词在文本中未出现:")
            if len(missing_words) <= 20:
                print(f"  {missing_words}")
            else:
                print(f"  前20个: {missing_words[:20]}")

        print("\n=== 词频统计（前15） ===")
        top_words = self.get_top_words(15)
        for i, (word, count) in enumerate(top_words):
            print(f"{i + 1:3d}. {word}: {count:5d} 次 ({100.0 * count / self.total_matches:.2f}%)")

        print("\n=== NPMI最高的20个词对 ===")
        sorted_npmi = self.get_sorted_npmi_results()
        limit = min(20, len(sorted_npmi))
        for i in range(limit):
            pair, npmi = sorted_npmi[i]
            co_count = self.pair_doc_count.get(pair, 0)
            count1 = self.word_doc_count.get(pair.first, 0)
            count2 = self.word_doc_count.get(pair.second, 0)
            print(
                f"{i + 1:3d}. {pair.first} - {pair.second}: NPMI={npmi:8.4f}, 共现={co_count:3d}, 文档频次=({count1},{count2})")

        if sorted_npmi:
            print("\n=== NPMI分布 ===")
            print(f"最高NPMI: {sorted_npmi[0][1]:.4f}")
            print(f"最低NPMI: {sorted_npmi[-1][1]:.4f}")

            # 统计正负NPMI
            positive_count = sum(1 for _, v in sorted_npmi if v > 0)
            negative_count = sum(1 for _, v in sorted_npmi if v < 0)
            zero_count = sum(1 for _, v in sorted_npmi if abs(v) < 0.0001)

            print(f"正NPMI词对: {positive_count} ({100.0 * positive_count / len(sorted_npmi):.1f}%)")
            print(f"负NPMI词对: {negative_count} ({100.0 * negative_count / len(sorted_npmi):.1f}%)")
            print(f"零NPMI词对: {zero_count} ({100.0 * zero_count / len(sorted_npmi):.1f}%)")

            # 计算NPMI平均值
            avg_npmi = sum(v for _, v in sorted_npmi) / len(sorted_npmi)
            print(f"平均NPMI: {avg_npmi:.4f}")

    def save_results_to_file(self, file_path):
        """保存NPMI结果到CSV文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("词1,词2,NPMI值,PMI值,共现文档数,词1文档频次,词2文档频次,词1概率,词2概率,联合概率\n")

            sorted_npmi = self.get_sorted_npmi_results()
            N = self.doc_count

            for pair, npmi in sorted_npmi:
                co_count = self.pair_doc_count.get(pair, 0)
                count1 = self.word_doc_count.get(pair.first, 0)
                count2 = self.word_doc_count.get(pair.second, 0)

                p1 = count1 / N
                p2 = count2 / N
                p12 = co_count / N

                # 计算PMI
                pmi = math.log(p12 / (p1 * p2)) if p1 * p2 > 0 and p12 > 0 else float('nan')

                f.write(
                    f"{pair.first},{pair.second},{npmi:.6f},{pmi:.6f},{co_count},{count1},{count2},{p1:.6f},{p2:.6f},{p12:.6f}\n")

        print(f"NPMI结果已保存到: {file_path} ({len(self.npmi_results)} 个词对)")

    def save_word_frequencies(self, file_path):
        """保存词频统计到CSV文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("词,出现次数,文档频次,文档概率\n")

            # 按总出现次数排序
            sorted_words = sorted(self.word_total_occurrences.items(),
                                  key=lambda x: x[1], reverse=True)

            N = self.doc_count

            for word, total_count in sorted_words:
                doc_count = self.word_doc_count.get(word, 0)
                doc_prob = doc_count / N

                f.write(f"{word},{total_count},{doc_count},{doc_prob:.6f}\n")

        print(f"词频统计已保存到: {file_path}")

    def save_detailed_report(self, file_path):
        """保存详细报告到TXT文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== NPMI分析报告 ===\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"诗歌数量: {self.doc_count}\n")
            f.write(f"文本总长度: {len(self.combined_text)} 汉字\n")
            f.write(f"词表大小: {len(self.word_list)} 个词\n")
            f.write(f"总匹配次数: {self.total_matches}\n")
            f.write(f"有共现的词对数: {len(self.npmi_results)}\n")
            f.write(f"计算耗时: {self.calculation_time * 1000:.2f} ms\n")
            f.write("注意：NPMI值域为[-1, 1]，1表示完全相关，-1表示完全不相关，0表示独立\n")

            f.write("\n=== 未出现的词 ===\n")
            missing = self.get_missing_words()
            if not missing:
                f.write("(所有词都至少出现一次)\n")
            else:
                f.write(f"共 {len(missing)} 个:\n")
                for i, word in enumerate(missing):
                    f.write(word)
                    if (i + 1) % 10 == 0:
                        f.write("\n")
                    elif i < len(missing) - 1:
                        f.write(", ")
                f.write("\n")

            f.write("\n=== 词频Top 50 ===\n")
            top_words = self.get_top_words(50)
            for i, (word, count) in enumerate(top_words):
                f.write(f"{i + 1:3d}. {word}: {count}\n")

            f.write("\n=== NPMI Top 100 ===\n")
            sorted_npmi = self.get_sorted_npmi_results()
            limit = min(100, len(sorted_npmi))
            for i in range(limit):
                pair, npmi = sorted_npmi[i]
                co_count = self.pair_doc_count.get(pair, 0)
                f.write(f"{i + 1:3d}. {pair.first} - {pair.second}: NPMI={npmi:.4f}, 共现={co_count}\n")

            # 添加NPMI统计摘要
            if sorted_npmi:
                f.write("\n=== NPMI统计摘要 ===\n")
                max_npmi = sorted_npmi[0][1]
                min_npmi = sorted_npmi[-1][1]
                avg_npmi = sum(v for _, v in sorted_npmi) / len(sorted_npmi)

                f.write(f"最高NPMI: {max_npmi:.4f}\n")
                f.write(f"最低NPMI: {min_npmi:.4f}\n")
                f.write(f"平均NPMI: {avg_npmi:.4f}\n")
                f.write(f"NPMI范围: [{min_npmi:.4f}, {max_npmi:.4f}]\n")

        print(f"详细报告已保存到: {file_path}")


# ==================== 主程序 ====================

def print_usage():
    """打印使用说明"""
    print("NPMI计算程序 - Python版本")
    print("=" * 50)
    print("\n说明：NPMI（标准化点间互信息）值域为[-1, 1]")
    print("      1表示完全相关，-1表示完全不相关，0表示独立\n")
    print("使用说明:")
    print("  python npmi_calculator.py <诗歌文件夹> <词表文件> [输出前缀]")
    print()
    print("参数说明:")
    print("  诗歌文件夹: 包含多首诗的文件夹（每首诗一个.txt文件）")
    print("  词表文件: 每行一个双字词的.txt文件")
    print("  输出前缀: 可选，输出文件的前缀（默认为'output_npmi'）")
    print()
    print("输出说明:")
    print("  NPMI值域为[-1, 1]，其中:")
    print("    1: 完全相关（总是同时出现）")
    print("    0: 相互独立（出现概率不相关）")
    print("    -1: 完全不相关（从不一起出现）")
    print()
    print("示例:")
    print("  python npmi_calculator.py ./poems/ word_list.txt result")
    print("  python npmi_calculator.py data/poems data/words.txt")


def run_demo():
    """运行演示示例"""
    print("\n正在运行演示示例...\n")

    # 创建演示数据
    demo_poems = [
        "春风又绿江南岸，明月何时照我还？",
        "春风化雨润江南，江南风光美如画。",
        "明月清风共此时，风光无限好风光。",
        "春风明月总相宜，江南风光最难忘。"
    ]

    demo_poem_names = ["诗1.txt", "诗2.txt", "诗3.txt", "诗4.txt"]
    demo_words = ["春风", "江南", "明月", "风光", "清风", "化雨", "何时", "相宜", "难忘"]

    # 创建临时文件
    demo_word_file = "demo_words.txt"
    with open(demo_word_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(demo_words))

    # 创建临时文件夹
    demo_folder = "demo_poems"
    os.makedirs(demo_folder, exist_ok=True)
    for name, content in zip(demo_poem_names, demo_poems):
        with open(os.path.join(demo_folder, name), 'w', encoding='utf-8') as f:
            f.write(content)

    try:
        run_analysis(demo_folder, demo_word_file, "demo_npmi")
    finally:
        # 清理临时文件
        os.remove(demo_word_file)
        for name in demo_poem_names:
            os.remove(os.path.join(demo_folder, name))
        os.rmdir(demo_folder)


def run_analysis(poems_folder, word_file, output_prefix):
    """
    执行NPMI分析

    Args:
        poems_folder: 诗歌文件夹路径
        word_file: 词表文件路径
        output_prefix: 输出文件前缀
    """
    print("正在读取文件...")
    print(f"诗歌文件夹: {poems_folder}")
    print(f"词表文件: {word_file}")

    # 1. 读取词表
    word_list = FileUtils.read_word_list(word_file)

    # 2. 读取所有诗歌
    poems, poem_names = FileUtils.read_poems_from_folder(poems_folder)

    if not poems:
        print("错误：未找到任何诗歌文件！")
        return

    print(f"诗歌数量: {len(poems)} 首")
    print(f"词表大小: {len(word_list)} 个词")

    # 3. 合并文本用于预处理信息（可选）
    combined_text = ''.join(poems)

    # 4. 检查词表
    if not word_list:
        print("错误：词表为空或没有有效的双字词！")
        return

    print(f"有效的双字词数量: {len(word_list)}")
    if len(word_list) <= 20:
        print(f"词表内容: {word_list}")
    else:
        print(f"前20个词: {word_list[:20]}")
        print(f"... 等 {len(word_list)} 个词")
    print()

    # 5. 计算NPMI
    print("开始计算NPMI...")
    calculator = PMICalculator(poems, word_list, poem_names)
    calculator.calculate()

    # 6. 显示统计信息
    calculator.print_statistics()

    # 7. 保存结果
    print("\n正在保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{output_prefix}_{timestamp}"

    calculator.save_results_to_file(f"{base_name}_npmi.csv")
    calculator.save_word_frequencies(f"{base_name}_freq.csv")
    calculator.save_detailed_report(f"{base_name}_report.txt")

    print("\n=== 文件输出完成 ===")
    print(f"1. NPMI结果: {base_name}_npmi.csv")
    print(f"2. 词频统计: {base_name}_freq.csv")
    print(f"3. 详细报告: {base_name}_report.txt")

    # 8. 保存清洗后的文本（可选）
    if len(combined_text) <= 10000:  # 只保存较小的文件
        cleaned_text = ''.join([TextPreprocessor.remove_line_breaks_only(p) for p in poems])
        FileUtils.save_to_file(cleaned_text, f"{base_name}_cleaned.txt")
        print(f"4. 清洗后文本: {base_name}_cleaned.txt")

    print("\n分析完成！")


def main():
    """主函数"""
    # # 检查参数
    # if len(sys.argv) < 3:
    #     print_usage()
    #     # 如果没有参数，运行演示示例
    #     run_demo()
    #     return

    # poems_folder = sys.argv[1]  # 诗歌文件夹路径
    # word_file = sys.argv[2]  # 词表文件路径
    # output_prefix = sys.argv[3] if len(sys.argv) > 3 else "output_npmi"  # 输出文件前缀

    poems_folder = r"H:\U-个人\个人\0-我的成果\我的论文\论文-22-古诗文情感AHCI\语料\按时期\盛唐"  # 诗歌文件夹路径
    word_file = r"LIST.txt"  # 词表文件路径
    output_prefix =  r"output_npmi_盛唐"  # 输出文件前缀
    try:
        run_analysis(poems_folder, word_file, output_prefix)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()