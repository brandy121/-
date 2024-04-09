import math
import os
import pickle
import jieba
import pandas as pd


def unigramCounts(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    return counts


def bigramCounts(words):
    counts = {}
    for i in range(len(words)-1):
        counts[(words[i], words[i+1])] = counts.get((words[i], words[i+1]), 0) + 1

    return counts


def trigramCounts(words):
    counts = {}
    for i in range((len(words)-2)):
        counts[(words[i], words[i+1], words[i+2])] = counts.get((words[i], words[i+1], words[i+2]), 0) + 1

    return counts


def entropyUnigram(counts):
    word_len = sum([item[1] for item in counts.items()])

    entropy = 0
    for word in counts.items():
        entropy -= word[1]/word_len * math.log(word[1] / word_len, 2)

    return entropy


def entropyBigram(uiCounts, biCounts):
    word_len = sum([item[1] for item in biCounts.items()])

    entropy = 0
    for word in biCounts.items():
        p_xy = word[1]/word_len
        p_x_y = word[1]/uiCounts[word[0][0]]
        entropy -= p_xy * math.log(p_x_y, 2)

    return entropy


def entropyTrigram(biCounts, triCounts):
    word_len = sum([item[1] for item in triCounts.items()])

    entropy = 0
    for word in triCounts.items():
        p_xy = word[1]/word_len
        p_x_y = word[1]/biCounts[(word[0][0], word[0][1])]
        entropy -= p_xy * math.log(p_x_y, 2)

    return entropy


if __name__ == '__main__':
    # 停词
    with open('ch_stop_word/cn_stopwords.pickle', 'rb') as f:
        extra_characters = pickle.load(f, encoding='utf-8')

    # extra_characters.append(' ')
    # extra_characters.append('\n')
    # extra_characters.append('\u3000')

    # 文件读取
    dirPath = "Chinese/inf.txt"
    txt = open(dirPath, "r", encoding='gb18030').read()
    names = txt.split(',')

    output = pd.DataFrame()
    for name in names:
        filePath = "Chinese/" + name + ".txt"
        txt = open(filePath, "r", encoding='gb18030').read()

        # 去除无用信息
        txt = txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')

        """ 字的信息熵 """
        # 去除无意义字符
        data = []
        for word in txt:
            if (word not in extra_characters) and (not word.isspace()):
                data.append(word)

        # 不同模型的词频统计
        uiCounts = unigramCounts(data)
        biCounts = bigramCounts(data)
        triCounts = trigramCounts(data)

        en_u = entropyUnigram(uiCounts)
        en_b = entropyBigram(uiCounts, biCounts)
        en_t = entropyTrigram(biCounts, triCounts)

        """ 词的信息熵 """
        # 分词
        cut_words = jieba.lcut(txt)

        # 去除无意义字符
        words = []
        for word in cut_words:
            if (word not in extra_characters) and (not word.isspace()):
                words.append(word)

        # 不同模型的词频统计
        uiCounts = unigramCounts(words)
        biCounts = bigramCounts(words)
        triCounts = trigramCounts(words)

        en_u_w = entropyUnigram(uiCounts)
        en_b_w = entropyBigram(uiCounts, biCounts)
        en_t_w = entropyTrigram(biCounts, triCounts)

        # 保存
        entropy = [en_u, en_b, en_t, en_u_w, en_b_w, en_t_w]
        output[name] = entropy
        print(entropy)

    output = output.T
    output.to_excel('entropy.xlsx')
