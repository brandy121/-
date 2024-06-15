import math
import os
import pickle
import random
import re
import time
from collections import Counter
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_novel_path = 'Chinese/天龙八部.txt'
char_key_dict_path = 'Chinese/chinese_3500.txt'
model_save_path = "./TextGener/transformer3_model.pkl"
model_save_path_pth = "./TextGener/transformer3.pth"
# 1 使用所有语料库，1-20，21-41
# 2 使用天龙八部 1-20，2-21
# 3 同2 去掉起始符和结束符：效果不错

use_gpu = torch.cuda.is_available()
print('torch.cuda.is_available() == ', use_gpu)
device = torch.device('cuda:0')


class Tokenizer:
    """
    词典编码器
    """
    UNKNOWN = "<unknown>"
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, tokens):
        # 补上特殊词标记：未知词标记、填充字符标记、开始标记、结束标记
        tokens = [Tokenizer.UNKNOWN, Tokenizer.PAD, Tokenizer.BOS, Tokenizer.EOS] + tokens
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {}  # 映射: 词 -> 编号
        self.id_token = {}  # 映射: 编号 -> 词
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word

        # 各个特殊标记的编号id，方便其他地方使用
        self.unknown_id = self.token_id[Tokenizer.UNKNOWN]
        self.pad_id = self.token_id[Tokenizer.PAD]
        self.bos_id = self.token_id[Tokenizer.BOS]
        self.eos_id = self.token_id[Tokenizer.EOS]

    def id_to_token(self, token_id):
        """
        编号 -> 词
        """
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        """
        词 -> 编号，取不到时给 UNKNOWN
        """
        return self.token_id.get(token, self.unknown_id)

    def encode(self, tokens):
        """
        词列表 -> <bos>编号 + 编号列表 + <eos>编号
        """
        # token_ids = [self.bos_id, ]  # 起始标记
        token_ids = []
        # 遍历，词转编号
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        # token_ids.append(self.eos_id)  # 结束标记
        return token_ids

    def decode(self, token_ids):
        """
        编号列表 -> 词列表(去掉起始、结束标记)
        """
        tokens = []
        for idx in token_ids:
            # 跳过起始、结束标记
            if idx != self.bos_id and idx != self.eos_id:
                tokens.append(self.id_to_token(idx))
        return tokens

    def __len__(self):
        return self.dict_size


class MyDataset(TensorDataset):

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length  # 每条数据的最大长度

    def __getitem__(self, index):
        line = self.data[index]
        word_ids = self.encode_pad_line(line)
        return torch.LongTensor(word_ids)

    def __len__(self):
        return len(self.data)

    def encode_pad_line(self, line):
        # 编码
        word_ids = self.tokenizer.encode(line)
        # 如果句子长度不足max_length，填充PAD
        word_ids = word_ids + [tokenizer.pad_id] * (self.max_length - len(word_ids))
        return word_ids


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来，这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, num_embeddings=4096, embedding_dim=128):
        super(TransformerModel, self).__init__()
        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model=embedding_dim, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=0)
        # 线性层输出需要和原始词典的字符编号范围对应
        self.predictor = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, src, tgt):
        # 生成 Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src).to(device)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt).to(device)

        # 词嵌入
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 喂数据给 Transformer
        # permute(1, 0, 2) 将src切换成“批次”在中间维度的形式，因为没有设置batch_first
        # tgt_mask = tgt_mask.type(torch.bool)
        # src_key_padding_mask = src_key_padding_mask.type(torch.bool)
        # tgt_key_padding_mask = tgt_key_padding_mask.type(torch.bool)

        out = self.transformer(src.permute(1, 0, 2), tgt.permute(1, 0, 2),
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        # 训练和推理时的行为不一样，在该模型外再进行线性层的预测，节约部分性能
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == Tokenizer.PAD] = float('-inf')
        return key_padding_mask


def load_words():
    """加载数据集"""
    # with open('NLPmodel/data_word_20.txt', encoding='utf8') as f:
    #     corpus_chars = f.read()
    #
    # with open('ch_stop_word/cn_stopwords.pickle', 'rb') as f:
    #     extra_characters = pickle.load(f, encoding='utf-8')
    # extra_characters = extra_characters[:63]
    # extra_characters.append('=')
    #
    # corpus_chars = corpus_chars.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
    # corpus_chars = corpus_chars.replace(' ', '')
    # corpus_chars = re.split(r'\n', corpus_chars)

    with open(train_novel_path, encoding='gb18030') as f:
        corpus_chars = f.read()

    with open('ch_stop_word/cn_stopwords.pickle', 'rb') as f:
        extra_characters = pickle.load(f, encoding='utf-8')
    extra_characters = extra_characters[:63]
    extra_characters.append('=')

    corpus_chars = corpus_chars.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
                                        '')
    data = []
    for word in corpus_chars:
        if (word not in extra_characters) and (not word.isspace()):
            data.append(word)

    corpus_chars = []
    sent_len = 20
    for i in range(len(data)-20):
        tmp = data[i:i+sent_len]
        corpus_chars.append(tmp)

    # 最小词频
    MIN_WORD_FREQUENCY = 8

    # 统计词频，利用Counter可以直接按单个字符进行统计词频
    counter = Counter()
    for line in corpus_chars:
        counter.update(line)
    # 过滤掉低词频的词
    tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]

    return corpus_chars, tokens


data, tokens = load_words()
# 实例化 Tokenizer
tokenizer = Tokenizer(tokens)
print("tokenizer_len: ", len(tokenizer))
#
# # 实例化MyDataset
# my_dataset = MyDataset(data, tokenizer, max_length=20)
# one_line_id = my_dataset[0].tolist()
# print("one_line_id: ", one_line_id)
#
# # 解码
# poetry_line = tokenizer.decode(one_line_id)
# print("poetry_line: ", "".join([w for w in poetry_line if w != Tokenizer.PAD]))
#
# # 读取一批数据，并解码
# temp_dataloader = DataLoader(dataset=my_dataset, batch_size=8, shuffle=True)
# one_batch_data = next(iter(temp_dataloader))
#
# for data_line_id in one_batch_data.tolist():
#     data_line = tokenizer.decode(data_line_id)
#     print("".join([w for w in data_line if w != Tokenizer.PAD]))

# 参数配置
EPOCH_NUM = 20  # 50
BATCH_SIZE = 64  # 64
DICT_SIZE = len(tokenizer)

# 数据
my_dataset = MyDataset(data, tokenizer, max_length=22)
train_dataloader = DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型
model = TransformerModel(num_embeddings=DICT_SIZE).to(device)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(1, EPOCH_NUM + 1):
    model.train()
    total_loss = 0
    data_progress = tqdm(train_dataloader, desc="Train...")
    for step, data in enumerate(data_progress, 1):
        start_time = time.time()

        data = data.to(device)
        # 随机选一个位置，拆分src和tgt
        e = random.randint(1, 10)
        src = data[:, :e]
        # tgt不要最后一个token，tgt_y不要第一个的token
        tgt, tgt_y = data[:, e:-1], data[:, e + 1:]

        # 进行Transformer的计算，再将结果送给最后的线性层进行预测
        out = model(src, tgt)
        out = model.predictor(out)
        # 使用view时，前面的数据必须是在内存连续的（即is_contiguous()为true）
        # 使用permute后，会导致数据不是内存连续的（即is_contiguous()为false），需要先调用contiguous()，才能继续使用view
        loss = criteria(out.view(-1, out.size(-1)), tgt_y.permute(1, 0).contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        print({'time': end_time-start_time})

        total_loss += loss.item()

        # 更新训练进度
        data_progress.set_description(f"Train... [epoch {epoch}/{EPOCH_NUM}, loss {(total_loss / step):.5f}]")

    torch.save(model, model_save_path)
    torch.save(model.state_dict(), model_save_path_pth)

""" 文本生成 """
# model = torch.load(model_save_path)
# model.eval()
# with torch.no_grad():
#     words = "跳舞"
#     for i in range(5):
#         word_ids = tokenizer.encode(words[-20:]+" ")
#         src = torch.LongTensor([word_ids[:-2]]).to(device)
#         tgt = torch.LongTensor([word_ids[-2:-1]]).to(device)
#         # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
#         for i in range(20):
#             out = model(src, tgt)
#             # 预测结果，只需最后一个词
#             predict = model.predictor(out[-1:])
#             # 找出最大值的index
#             y = torch.argmax(predict, dim=2)
#             # 和之前的预测结果拼接到一起
#             tgt = torch.cat([tgt, y], dim=1)
#
#         tgt_decode = "".join([w for w in tokenizer.decode(tgt[0].tolist()) if w != Tokenizer.PAD])
#         words = words + tgt_decode[1:]
#
#     print(words)

