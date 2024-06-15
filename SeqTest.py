# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:39:57 2021
@author: Administrator
"""

import argparse
import pickle

import torch
from torch import nn
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
import datetime
import math

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_novel_path = 'Chinese/天龙八部.txt'
char_key_dict_path = 'Chinese/chinese_3500.txt'
model_save_path = "./TextGener/seq_model.pkl"
model_save_path_pth = "./TextGener/seq.pth"

use_gpu = torch.cuda.is_available()
print('torch.cuda.is_available() == ', use_gpu)
device = torch.device('cuda:0')


def dictGet(dict1, index):
    length1 = len(dict1)

    if index >= 0 and index < length1:
        return dict1[index]
    else:
        return dict1[0]


def dictGetValue(dict1, indexZifu):
    if indexZifu in dict1:
        return dict1[indexZifu]
    else:
        return dict1['*']


def getNotSet(list1):
    '''
    返回一个新列表,如何删除列表中重复的元素且保留原顺序
    例子
        list1 = 1 1 2 3 3 5
        return 1 2 3 5
    '''
    l3 = []
    for i in list1:
        if i not in l3:
            l3.append(i)
    return l3;


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.words = self.load_words()

        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # self.words_list = list( self.words )

        # 把小说的 字 转换成 int
        self.words_indexes = []

        # 把字典里没有的字符 用'*'表示，也就是Chinese_characters_3500.txt没有的字符
        for w in self.words:
            if (w in self.word_to_index) == False:
                self.words_indexes.append(1482)  # 1482 =='*'
                # print(w,'= *',)
            else:
                self.words_indexes.append(self.word_to_index[w])
                # print(w,'= ',self.word_to_index[w])

    def load_words(self):
        """加载数据集"""
        with open(train_novel_path, encoding='gb18030') as f:
            corpus_chars = f.read()

        with open('ch_stop_word/cn_stopwords.pickle', 'rb') as f:
            extra_characters = pickle.load(f, encoding='utf-8')
        extra_characters = extra_characters[:63]
        extra_characters.append('=')

        corpus_chars = corpus_chars.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = []
        for word in corpus_chars:
            if (word not in extra_characters) and (not word.isspace()):
                data.append(word)

        data = data
        print('length', len(data))
        # corpus_chars = corpus_chars[0:10000]
        return data

    def get_uniq_words(self):
        with open(char_key_dict_path, 'r', encoding='utf-8') as f:
            text = f.read()
        idx_to_char = list(text)  # 不能使用 set(self.words) 函数 ,因为每次启动随机,只能用固定的
        return idx_to_char

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.args.sequence_length]).cuda(),
            torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]).cuda(),
            # torch.tensor(self.words_indexes[index + 1:index + 2]).cuda()
        )


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.input_size = 128
        self.hidden_size = 256
        self.embedding_dim = self.input_size
        self.num_layers = 2

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).cuda()
        self.fc = nn.Linear(self.hidden_size, n_vocab).cuda()

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda())


class Seq2seq(nn.Module):
    def __init__(self, dataset):
        super(Seq2seq, self).__init__()
        self.input_size = 128
        self.hidden_size = 256
        self.embedding_dim = self.input_size
        self.num_layers = 2

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=n_vocab, num_layers=self.num_layers)

    def forward(self, x, h, c):
        embed = self.embedding(x)
        output, (h, c) = self.encoder(embed, (h, c))
        logits, (_, _) = self.decoder(output)
        # logits = self.fc(output)

        return logits, h, c

    def init_state(self, sequence_length):
        h = torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda()
        return h, c


def train(dataset, model, args):
    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)  # 1e-3

    for epoch in range(args.max_epochs):
        """ Seq2Seq """
        h, c = model.init_state(args.sequence_length)
        # state = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            start_time = time.time()

            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()

            """ Seq2Seq """
            y_pred, h, c = model(x, h, c)
            h = h.detach()
            c = c.detach()
            # y_pred, state = model(x, state)
            # state = state.detach()

            loss = criterion(y_pred.transpose(1, 2), y)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            end_time = time.time()

            if batch % 100 == 0:
                torch.save(model, model_save_path)
                torch.save(model.state_dict(), model_save_path_pth)
                print({'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'time': end_time-start_time})


def predict(dataset, model, text, next_words=20):
    # words = text.split(' ')
    words = list(text)
    model.eval()

    device = 'cuda:0'
    model.to(device)

    """ Seq2Seq """
    # h, c = model.init_state(len(text))
    state = model.init_state(len(text))

    for i in range(0, next_words):
        x = torch.tensor([[dictGetValue(dataset.word_to_index, w) for w in words[i:]]]).cuda()
        # y_pred, h, c = model(x, h, c)
        # h = h.detach()
        # c = c.detach()
        y_pred, state = model(x, state)
        state = state.detach()

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        # p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        # p = torch.from_numpy(p).cuda(0)
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dictGet(dataset.index_to_word, word_index))

    return "".join(words)


parser = argparse.ArgumentParser(description='rnn')
parser.add_argument('--max-epochs', type=int, default=10)  # default=20
parser.add_argument('--batch-size', type=int, default=128)  # default=256
parser.add_argument('--sequence-length', type=int, default=20)  # sequence-length 每次训练多长的句子, default=20
args = parser.parse_args([])

dataset = Dataset(args)

""" 训练 """
model = Seq2seq(dataset)
# model = Model(dataset)
train(dataset, model, args)

torch.save(model, model_save_path)
torch.save(model.state_dict(), model_save_path_pth)
print("训练完成")

""" 预测 """
# model = torch.load(model_save_path)
# device = 'cuda:0'
# model.to(device)
#
# pred_novel_start_text = ('跳舞')
# pred = predict(dataset, model, pred_novel_start_text, 54)
# print(pred)
