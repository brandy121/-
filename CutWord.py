import os
import pickle

import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter


def cutWord():
    # 停词
    with open('ch_stop_word/cn_punctuation.pickle', 'rb') as f:
        extra_characters = pickle.load(f, encoding='utf-8')

    extra_characters.append(' ')
    extra_characters.append('\n')
    extra_characters.append('\u3000')

    # 文件读取
    dirPath = "Chinese/inf.txt"
    txt = open(dirPath, "r", encoding='gb18030').read()
    names = txt.split(',')

    for name in names:
        filePath = "Chinese/" + name + ".txt"
        txt = open(filePath, "r", encoding='gb18030').read()

        # 去除无用信息
        txt = txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')

        # 分词
        words = jieba.lcut(txt)

        # 统计词频
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        # 去除无意义字符
        for word in extra_characters:
            if word in counts:
                del counts[word]

        # 保存分词后的字典文件
        savePath = 'ch_cut_word/' + name + ".pickle"
        with open(savePath, 'wb') as f:
            pickle.dump(counts, f)

        # 排序
        items = list(counts.items())
        items.sort(key=lambda x: x[1], reverse=True)
        sort_list = sorted(counts.values(), reverse=True)

        # 验证zipf-law
        figPath = 'cut_word_fig/' + name + ".jpg"
        x = [i for i in range(len(sort_list))]
        plt.plot(x, sort_list)

        plt.title(name, fontsize=18)
        plt.xlabel('rank', fontsize=18)
        plt.ylabel('freq', fontsize=18)
        plt.yticks([pow(10, i) for i in range(0, 4)])
        plt.xticks([pow(10, i) for i in range(0, 4)])
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(figPath)
        plt.show()


def stop_words():
    dirPath = 'ch_stop_word'
    names = os.listdir(dirPath)

    for name in names:
        path = os.path.join(dirPath, name)
        txt = open(path, "r", encoding='utf-8').read()
        txt = txt.split('\n')

        savePath = 'ch_stop_word/' + name[:-4] + ".pickle"
        with open(savePath, 'wb') as f:
            pickle.dump(txt, f)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 获取停词
    # stop_words()

    # 每个文件的词频
    # cutWord()

    # 总词频图
    dirPath = "ch_cut_word"
    names = os.listdir(dirPath)

    counts_combined = {}
    # 遍历每个文件
    for file_name in names:
        with open(os.path.join(dirPath, file_name), 'rb') as f:
            counts = pickle.load(f, encoding='utf-8')

        # 将每个字典中的计数合并到一个总字典中
        for word, count in counts.items():
            if word in counts_combined:
                counts_combined[word] += count
            else:
                counts_combined[word] = count

    # 排序
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    sort_list = sorted(counts.values(), reverse=True)

    # 验证zipf-law
    x = [i for i in range(len(sort_list))]
    plt.plot(x, sort_list)

    plt.title('Zipf-Law', fontsize=18)
    plt.xlabel('rank', fontsize=18)
    plt.ylabel('freq', fontsize=18)
    plt.yticks([pow(10, i) for i in range(0, 4)])
    plt.xticks([pow(10, i) for i in range(0, 4)])
    plt.yscale('log')
    plt.xscale('log')

    plt.show()