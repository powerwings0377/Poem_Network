# 把 your_file.txt 换成你自己的文件名
with open("全唐诗.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("总字符数：", len(text))