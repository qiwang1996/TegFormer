from model import Topic2Essay
import torch
import time
import random
import numpy as np
from dataloader import Dataloader, TextTokenizer

# coding: UTF-8
import os
import math
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import torch.nn.functional as F

if __name__ == '__main__':
    # parser.add_argument('--gpt2_model', default='train', help='action')

    gpt2_model_path = './model/gpt2'
    data_file = './data/test.txt'
    save_model_path = './model/trained_model/'
    bert_path = './model/bert'
    topic_file = './data/topic_dict.txt'
    
    write_gen_file = './data/gen.txt'


    device= [0]
    batch_size = 200
    vocab_size = 21128
    topic_vocab_size = 7993

    text_pad_id = 0
    text_pad_size = 130
    topic_pad_id = 7993
    topic_pad_size = 5
    model = Topic2Essay(gpt2_model_path, vocab_size, topic_vocab_size).cuda()

    pretrained_dict = torch.load(save_model_path + 'pretrain.pkl') # 预训练模型参数保存地址
    model_dict = model.state_dict()  # 自己的模型参数变量

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 去除一些不需要的参数
    model_dict.update(pretrained_dict)  # 参数更新
    model.load_state_dict(model_dict)  # 加载

    model.eval()

    data_loader = Dataloader(data_file, topic_file, bert_path, text_pad_id, text_pad_size,
                             topic_pad_id, topic_pad_size, batch_size)

    max_len = 150
    text_tokenizer = TextTokenizer(bert_path, 0, 130)
    start_symbol = text_tokenizer.tokenizer.convert_tokens_to_ids('[CLS]')
    
    wfile = open(write_gen_file, 'w+')
    for batch_i in tqdm(range(data_loader.batch_num)):
        batch = data_loader.get_batch()
        src, src_mask = batch.src.cuda(), batch.src_mask.cuda()
        memory = model.encode(src, src_mask).cuda()
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.LongTensor).cuda()
        for i in range(max_len - 1):
            ys = model.decode(memory, src_mask, ys)
        ys = ys.cpu().detach().numpy().tolist()
        for sent in ys:
            sent_token = []
            for word in sent:
                token = text_tokenizer.tokenizer.convert_ids_to_tokens(word)
                sent_token.append(token)
                if token == '[SEP]':
                    break
            wfile.write(''.join(sent_token) + '\n')
    wfile.close()


