from model import Topic2Essay
import torch
import time
import random
from transformer_model import MultiGPULossCompute, LabelSmoothing
import numpy as np
from dataloader import Dataloader, Batch

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
    data_file = './data/zhihu.txt'
    save_model_path = './model/trained_model/'
    bert_path = './model/bert'
    topic_file = './data/topic_dict.txt'


    epoch = 40
    batch_size = 128

    vocab_size = 21128
    topic_vocab_size = 7393

    text_pad_id = 0
    text_pad_size = 130
    topic_pad_id =  0
    topic_pad_size = 5

    device= [0, 1]
    model = Topic2Essay(gpt2_model_path, vocab_size, topic_vocab_size) 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothing(vocab_size, 0, 0.1).cuda()
    data_loader = Dataloader(data_file, topic_file, bert_path, text_pad_id, text_pad_size,
                             topic_pad_id, topic_pad_size, batch_size, True)

    for epoch_i in range(epoch):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0

        # model_par = torch.nn.DataParallel(model, device_ids=device)
        model.train()
        n_batches = data_loader.batch_num

        for batch_i in range(n_batches):
            batch = data_loader.get_batch()
            out = model(batch.src.cuda(0), batch.src_mask.cuda(0),
                        batch.trg.cuda(0), batch.trg_mask.cuda(0))

            #             if (out <= 0).sum() != 0:
            #                 raise Exception('out nan')

            # trg_y = batch.trg_y.unsqueeze(2).cuda()

            # loss = torch.gather(-torch.log(out), 2, trg_y)
            # loss = loss[batch.trg_mask.cuda()].mean()

            # bs, seq_len = trg_y.size(0), trg_y.size(1)
            loss_compute = MultiGPULossCompute(criterion, devices=device, opt=optimizer)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            # loss = loss[batch.trg_mask.cuda()].mean()
            #             if np.isnan(loss.cpu().detach().numpy()):
            #                 raise Exception('loss nan')

            total_loss += loss.item()
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if batch_i % 4 == 0:
                elapsed = time.time() - start
                print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                      (epoch_i, batch_i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        torch.save(model.state_dict(), save_model_path + 'pretrain.pkl')
        print("Epoch %d  Loss: %f" % (epoch_i, total_loss / total_tokens))
