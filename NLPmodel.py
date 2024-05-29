import math
import pickle
import jieba
import multiprocessing
from gensim.models import KeyedVectors, word2vec, Word2Vec
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
from gensim.scripts.glove2word2vec import glove2word2vec


def ceshi():
    sentences_list = [
        '详细了解园区规划，走访入驻企业项目，现场察看产品研发和产业化情况。他强调，',
        '要坚持规划先行，立足高起点、高标准、高质量，科学规划园区组团，提升公共服务水平，',
        '注重产城融合发展。要增强集聚功能，集聚产业、人才、技术、资本，加快构建从基础研究、',
        '技术研发、临床实验到药品生产的完整产业链条，完善支撑产业发展的研发孵化、成果转化、',
        '生产制造平台等功能配套，推动产学研用协同创新，做大做强生物医药产业集群。唐良智在调研',
        '中指出，我市生物医药产业具有良好基础，但与高质量发展的要求相比，在规模、结构、创新能力',
        '等方面还存在不足。推动生物医药产业高质量发展，努力培育新兴支柱产业，必须紧紧依靠创新创业',
        '创造，着力营造良好发展环境。要向改革开放要动力，纵深推进“放管服”改革，用好国际国内创新资源，',
        '大力引进科技领军人才、高水平创新团队。要坚持问题导向，聚焦企业面临的困难和问题，把握生物医药产业',
        '发展特点，精准谋划、不断完善产业支持政策，切实增强企业获得感。要精准服务企业，构建亲清新型政商关系，',
        '以高效优质服务助力企业发展',
        '2018年我省软件和信息服务业发展指数为67.09，']
    # 加载停用词表
    sentences_cut = []
    # 结巴分词
    for ele in sentences_list:
        cuts = jieba.cut(ele, cut_all=False)
        new_cuts = []
        for cut in cuts:
            new_cuts.append(cut)
        sentences_cut.append(new_cuts)

    # 分词后的文本保存在data.txt中
    with open('data.txt', 'w') as f:
        data_str = '\n'.join([' '.join(row) for row in sentences_cut])
        f.write(data_str)


def get_words(cutIsTrue):
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

    words = []
    sent_len = 100
    for name in names:
        filePath = dir + name + ".pickle"
        with open(filePath, 'rb') as f:
            data = pickle.load(f)

        # 分句
        words_len = len(data)
        sent_num = math.ceil(words_len/sent_len)
        for i in range(sent_num):
            if i == sent_num-1:
                tmp = data[i*sent_len:-1]
            else:
                tmp = data[i*sent_len:(i+1)*sent_len]
            words.append(tmp)

    with open('NLPmodel/data.txt', 'w', encoding='utf-8') as f:
        data_str = '\n'.join([' '.join(row) for row in words])
        f.write(data_str)


def model_word2vec():

    # 训练
    sentences = list(word2vec.LineSentence('NLPmodel/data.txt'))

    model = Word2Vec(sentences, min_count=5, window=5,
                     sg=0, workers=multiprocessing.cpu_count())
    model.save('NLPmodel/word2vec.model')

    # 读取模型
    model = Word2Vec.load('NLPmodel/word2vec.model')
    # print(model)

    # 词语聚类
    # for key in model.wv.similar_by_word('客栈', topn=10):
    #     print(key)

    # 语意距离
    print(model.wv.similarity('杨过', '小龙女'))
    print(model.wv.similarity('杨过', '东方'))
    print(model.wv.similarity('华山派', '弟子'))
    print(model.wv.similarity('华山派', '剑法'))

    # t-SNE
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #
    # vector = model.wv.vectors[:100]
    # label = model.wv.index_to_key[:100]
    #
    # tsne = TSNE(n_components=2)
    # vector2 = tsne.fit_transform(vector)
    #
    # x_vals = [v[0] for v in vector2]
    # y_vals = [v[1] for v in vector2]  # 创建一个 trace
    # trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=label)
    # data = [trace]
    #
    # plot(data, filename='word-embedding-plot.html')

    pass


def model_glove():
    dir = 'E:\\A_张海尼文件\\课程学习\\自然语言处理\\作业3\\GloVe-master\\'
    glove_file = dir + 'vectors.txt'
    glove_word = dir + 'w2v.txt'  # 输出文件

    # 转换
    # glove2word2vec(glove_file, glove_word)
    # 加载模型
    model = KeyedVectors.load_word2vec_format(glove_word)
    # print(model)

    # 词语聚类
    # for key in model.similar_by_word('杨过', topn=10):
    #     print(key)

    # 语意距离
    print(model.similarity('杨过', '小龙女'))
    print(model.similarity('杨过', '东方'))
    print(model.similarity('华山派', '弟子'))
    print(model.similarity('华山派', '剑法'))

    # t-SNE
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #
    # vector = model.vectors[:100]
    # label = model.index_to_key[:100]
    #
    # tsne = TSNE(n_components=2)
    # vector2 = tsne.fit_transform(vector)
    #
    # x_vals = [v[0] for v in vector2]
    # y_vals = [v[1] for v in vector2]  # 创建一个 trace
    # trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=label)
    # data = [trace]
    #
    # plot(data, filename='NLPmodel/word-glove-plot.html')

    pass


if __name__ == '__main__':
    # ceshi()
    # get_words(True)
    model_word2vec()
    # model_glove()
