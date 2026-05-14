import matplotlib.pyplot as plt
import numpy as np

# 替换成你的txt文件路径
txt_path = "output_BPE_original.txt"

words = []
freqs = []

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        word, num = line.split()
        words.append(word)
        freqs.append(int(num))

# 词语序号作为x
x = np.arange(1, len(words) + 1)

plt.figure(figsize=(16, 6))
plt.plot(x, freqs, marker="o", linewidth=2, markersize=3, color="#2E86AB")

# 全部英文
plt.xlabel("Word Index", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Word Frequency Distribution", fontsize=14)

# X轴刻度 间隔1000
x_max = len(words)
xticks = np.arange(0, x_max + 1000, 1000)
plt.xticks(xticks)

plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()