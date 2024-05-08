import csv
import math
import os
import pickle
import jieba
import pandas as pd
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import pyLDAvis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def get_data(cutIsTrue):
    # 停词
    with open('ch_stop_word/cn_stopwords.pickle', 'rb') as f:
        extra_characters = pickle.load(f, encoding='utf-8')

    # 文件读取
    dirPath = "Chinese/inf.txt"
    txt = open(dirPath, "r", encoding='gb18030').read()
    names = txt.split(',')

    for name in names:
        filePath = "Chinese/" + name + ".txt"
        txt = open(filePath, "r", encoding='gb18030').read()

        # 去除无用信息
        txt = txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')

        data = []
        if cutIsTrue is False:
            dir = "ch_corpus_word/"

            # 去除无意义字符
            for word in txt:
                if (word not in extra_characters) and (not word.isspace()):
                    data.append(word)
        else:
            dir = "ch_corpus_words/"

            # 分词
            cut_words = jieba.lcut(txt)

            # 去除无意义字符
            for word in cut_words:
                if (word not in extra_characters) and (not word.isspace()):
                    data.append(word)

        savePath = dir + name + ".pickle"
        with open(savePath, 'wb') as f:
            pickle.dump(data, f)


def get_dataset(cutIsTrue, K):
    if cutIsTrue is False:
        dir = "ch_corpus_word/"
        save_dir = "ch_dataset/word"
    else:
        dir = "ch_corpus_words/"
        save_dir = "ch_dataset/words"

    # 文件读取 ---------------------------------------------------------
    dirPath = "Chinese/inf.txt"
    txt = open(dirPath, "r", encoding='gb18030').read()
    names = txt.split(',')

    dict = {}  # 字典，以 name 为索引
    word_len = 0  # 总词数
    for name in names:
        filePath = dir + name + ".pickle"
        with open(filePath, 'rb') as f:
            data = pickle.load(f)

        word_len += len(data)
        dict[name] = data

    # 抽取段落数 ---------------------------------------------------------
    id = 1
    con_list = []
    count_sum = 0
    # print("名称:抽取段落数----总词数")
    for name in names:
        count = math.floor(len(dict[name])/word_len * 1000 + 0.48)
        print(f"{name}:{count}----{len(dict[name])}")

        # 抽取段落
        pos = int(len(dict[name])//count)
        for i in range(count):
            tmp = dict[name][(i*pos):(i*pos+K)]
            con = {
                'id': id,
                'name': name,
                'data': tmp
            }
            con_list.append(con)
            id += 1

    savePath = save_dir + "_" + str(K) + ".csv"
    with open(savePath, 'a', newline='', encoding='utf_8') as f:
        csv_header = ['id', 'name', 'data']
        csv_writer = csv.DictWriter(f, csv_header)
        if f.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(con_list)


def lda(cutIsTrue, K, T):
    if cutIsTrue is False:
        file_dir = "ch_dataset/word"
        save_dir = "lda_model/word"

    else:
        file_dir = "ch_dataset/words"
        save_dir = "lda_model/words"

    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    # 一个主题可以由词汇分布表示，一个段落可以由主题分布表示
    # 在所有段落上建模
    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)

    # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.filter_extremes(no_below=20, no_above=0.9)
    dict.compactify()  # 去掉因删除词汇而出现的空白

    corpus = [dict.doc2bow(text) for text in dataList]  # 表示为第几个单词出现了几次

    # LDA模型
    print(f'cutIsTrue: {cutIsTrue}   K:{K}  T:{T}  ----------------')
    ldamodel = LdaModel(corpus, num_topics=T, id2word=dict, passes=40, random_state=6)  # 分为T个主题
    # 字：passes=40, random_state=6

    # 模型评估
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=dataList, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    per = ldamodel.log_perplexity(corpus)
    print(f'perplexity:{per}     coherence:{coherence_lda}')

    # 提取主题
    topics = ldamodel.show_topics(num_words=T)
    # 输出主题
    for topic in topics:
        print(topic)

    # 保存模型 -----------------------------------------------------------------
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # dict.save(savePath + "_dict.dict")
    # corpora.MmCorpus.serialize(savePath + "_corpus.mm", corpus)
    ldamodel.save(savePath + "_ldaModel.model")


def lda_test(cutIsTrue, K, T):
    if cutIsTrue is False:
        file_dir = "ch_dataset/word"
        save_dir = "lda_model/word"
    else:
        file_dir = "ch_dataset/words"
        save_dir = "lda_model/words"

    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)

    dict.filter_extremes(no_below=20, no_above=0.9)  # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.compactify()  # 去掉因删除词汇而出现的空白

    corpus = [dict.doc2bow(text) for text in dataList]  # 表示为第几个单词出现了几次

    # 加载模型
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # 加载 LDA 模型
    lda = LdaModel.load(savePath + "_ldaModel.model")

    # 评估模型性能
    coherence_model_lda = CoherenceModel(model=lda, texts=dataList, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    per = lda.log_perplexity(corpus)
    print(f'perplexity:{per}     coherence:{coherence_lda}')

    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dict)
    pyLDAvis.show(vis, local=False)


def classify(cutIsTrue, K, T):
    if cutIsTrue is False:
        file_dir = "ch_dataset/word"
        save_dir = "lda_model/word"
    else:
        file_dir = "ch_dataset/words"
        save_dir = "lda_model/words"

    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)
    dict.filter_extremes(no_below=20, no_above=0.9)  # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.compactify()  # 去掉因删除词汇而出现的空白

    # 加载模型
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # 加载 LDA 模型
    lda = LdaModel.load(savePath + "_ldaModel.model")

    # 将每个段落进行做主题分布
    topic_matrix = []
    for tmp in dataList:
        cor = dict.doc2bow(tmp)

        topic_distribution = lda.get_document_topics(cor,  minimum_probability=0)
        topic_distribution = [prob for topic, prob in topic_distribution]

        topic_matrix.append(topic_distribution)

    topic_matrix = np.array(topic_matrix)

    # 获取标签
    label = data.iloc[:, -2].tolist()
    label = np.array(label)

    label_encoder = LabelEncoder()
    label_num = label_encoder.fit_transform(label)

    # 训练集训练分类模型
    # 将标签和topic对应，然后划分数据集，进行分类
    result = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(topic_matrix, label_num, test_size=100, random_state=i*20)
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=0)
        # 字：n_estimators=100, max_depth=12, random_state=0
        clf.fit(X_train, y_train)

        x_pred = clf.predict(X_train)
        x_acc = accuracy_score(y_train, x_pred)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        F1 = 2*precision*recall/(precision+recall)
        tmp_result = [accuracy, precision, recall, F1]

        print("-----------------------------------------")
        print(f"{i}:训练集准确率：{x_acc}")
        print(f"{i}:测试集评价：{tmp_result}")

        result.append(tmp_result)

    result = np.array(result)
    print(np.mean(result, axis=0))


if __name__ == '__main__':
    cutIsTrue = True

    # 创建语料库
    # get_data(cutIsTrue)

    # 抽取数据集，一共抽取1000个段落，每个段落 K 个token（20,100,500,1000,3000）
    # 段落的标签是对应小说
    K = 3000
    # get_dataset(cutIsTrue, K)

    # LDA 文本建模，主体数量为T
    T = 50
    # lda(cutIsTrue, K, T)
    lda_test(cutIsTrue, K, T)

    # 根据主体分布进行分类
    # 10 次交叉验证
    # classify(cutIsTrue, K, T)

    # pandas                       2.0.3



