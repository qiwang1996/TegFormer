import torch
import json


class GatBaseEle:
    def __init__(self, kw_looking_table, relation_json_file_path, kw_pad_index=2870, kw_cnt=8):
        with open(relation_json_file_path, 'r+') as fo:
            realtion_dict = json.load(fo)

        self.kw_looking_table = kw_looking_table
        self.relation = [[] for _ in range(len(realtion_dict))]

        for node, neibors in realtion_dict.items():
            node = int(node)
            self.relation[node] = [int(i) for i in neibors]
            self.relation[node] = (self.relation[node] + [kw_pad_index] * kw_cnt)[:kw_cnt]
            
        self.relation = torch.LongTensor(self.relation).cuda(0)
        self.mask = self.relation != kw_pad_index

    def get_neibor_embed(self, src):
        neibor = self.relation[src]  # batch_size, topic_num, neibor_num
        mask = self.mask[src]  # batch_size, topic_num, neibor_num
        neibor_embedding = self.kw_looking_table(neibor)  # batch_size, topic_num, neibor_num, word_dim
        return neibor_embedding, mask


class Gat(torch.nn.Module):
    def __init__(self, word_dim, h=4):
        super(Gat, self).__init__()
        self.query_layer = torch.nn.Linear(word_dim, word_dim)
        self.key_layer = torch.nn.Linear(word_dim, word_dim)
        self.val_layer = torch.nn.Linear(word_dim, word_dim)
        self.h = h

    def forward(self, neibor_embedding, mask, x):
    
        # src:  batch_size, topic_num
        k = self.key_layer(neibor_embedding)  # batch_size, topic_num, neibor_num, word_dim
        nerbor_num = k.size(2)
        q = self.query_layer(x)  # batch_size, topic_num, word_dim
        q = q.unsqueeze(2).repeat(1, 1, nerbor_num, 1)  # batch_size, topic_num, neibor_num, word_dim
        v = self.val_layer(neibor_embedding)  # batch_size, topic_num, neibor_num, word_dim
        # q,k,v split to multihead
        h = self.h

        q = q.view(q.size(0), q.size(1), q.size(2), h, q.size(3) // h)
        # batch_size, topic_num, neibor_num, h, word_dim //h
        k = k.view(k.size(0), k.size(1), k.size(2), h, k.size(3) // h)
        v = v.view(v.size(0), v.size(1), v.size(2), h, v.size(3) // h).transpose(2, 3)

        ret = q * k  # batch_size, topic_num, neibor_num, h, word_dim //h
        ret = torch.sum(ret, 4)  # batch_size, topic_num, neibor_num, h
        ret = ret.transpose(2, 3)  # batch_size, topic_num,  h,  neibor_num

        mask = mask.unsqueeze(2).repeat(1, 1, h, 1)  # batch_size, topic_num, h, neibor_num
        ret = ret.masked_fill(mask == 0, -1e9)  # batch_size, topic_num, h, neibor_num

        scores = torch.softmax(ret, dim=3)  # batch_size, topic_num, h, neibor_num
        scores = scores.unsqueeze(4)  # batch_size, topic_num, h, neibor_num, 1
        vec = scores * v  # batch_size, topic_num, h, neibor_num, word_dim //h
        vec = torch.sum(vec, dim=3)  # batch_size, topic_num, h, word_dim //h
        agg_embedding = vec.view(vec.size(0), vec.size(1), -1)  # batch_size, topic_num, word_dim
        return agg_embedding