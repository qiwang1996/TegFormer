import torch

from transformer_model import Transformer, subsequent_mask
from transformers import GPT2LMHeadModel

from torch.nn import DataParallel
from datetime import datetime

import os
import tqdm
import random


class Topic2Essay(torch.nn.Module):
    def __init__(self, gpt2_model_path, vocab_size, topic_vocab_size, kw_vocab_size=2870):
        super(Topic2Essay, self).__init__()
        relation_json_file_path = './GNN/relations.json'
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).cuda(0)
        self.topic_looking_table = torch.nn.Embedding(topic_vocab_size,
                                                512).cuda(0)
        self.kw_looking_table = torch.nn.Embedding(kw_vocab_size + 1,
                                                512).cuda(0)     
        self.tgt_looking_table = torch.nn.Embedding(vocab_size,
                                        32).cuda(0)     
                                        
        self.up_proj = torch.nn.Linear(32, 768).cuda(0)
                                                                                   
        for param in self.gpt2_model.parameters():
            param.requires_grad = False
        self.plugin_transformer = Transformer(tgt_vocab=vocab_size,
                                              relation_json_file_path=relation_json_file_path,
                                              topic_looking_table=self.topic_looking_table,
                                              kw_looking_table=self.kw_looking_table,
                                              Encoder_N=6, Decoder_N=6,
                                              d_model=512, d_ff=2048, h=8, dropout=0.1).cuda(0)
        for param in self.plugin_transformer.model.parameters():
            param.requires_grad = True
        self.trans_dim_net = torch.nn.Linear(768, 512).cuda(0)

    def forward(self, src, src_mask, trg, trg_mask):
        src_embed = self.topic_looking_table(src)
        outputs = self.gpt2_model(trg, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        
        tgt_hidden_states = self.up_proj(self.tgt_looking_table(trg))
        
        src_mask = src_mask.unsqueeze(-2)
        attn_score = self.gate(torch.cat([last_hidden_states,
                                          tgt_hidden_states], dim=2))
        # bs, seq_len, 1

        input_embed = attn_score * last_hidden_states + \
                      (1 - attn_score) * tgt_hidden_states
        trg_embed = self.trans_dim_net(input_embed)
        out = self.plugin_transformer(src.cuda(0), src_embed.cuda(0), src_mask.cuda(0), trg_mask.cuda(0), trg_embed.cuda(0))
        plugin_logits = self.plugin_transformer.model.generator(out)
        return plugin_logits

    def encode(self, src, src_mask):
        src_embed = self.topic_looking_table(src)
        src_mask = src_mask.unsqueeze(-2)
        memory = self.plugin_transformer.model.encode(src, src_embed, src_mask)
        return memory

    # memory, src_mask, tgt_embed, tgt_mask
    def decode(self, memory, src_mask, ys, k_value=10, eos_id = 102):
        src_mask = src_mask.unsqueeze(-2)
        trg = ys
        trg_mask = subsequent_mask(ys.size(1))
        trg_mask = trg_mask.expand([ys.size(0), trg_mask.size(1), trg_mask.size(1)]).cuda()
        
        tgt_hidden_states = self.up_proj(self.tgt_looking_table(trg))

        outputs = self.gpt2_model(trg, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        attn_score = self.gate(torch.cat([last_hidden_states,
                                          tgt_hidden_states], dim=2))
        #bs, seq_len, 1
        input_embed = attn_score * last_hidden_states + \
                      (1 - attn_score) * tgt_hidden_states
        trg_embed = self.trans_dim_net(input_embed)

        out = self.plugin_transformer.model.decode(memory, src_mask, trg_embed, trg_mask)
        plugin_logits = self.plugin_transformer.model.generator(out[:, -1])
        logits = torch.exp(plugin_logits)

        topk_prob_, topk_index_ = torch.sort(logits, dim=1, descending=True)
        topk_prob = topk_prob_[:,:k_value]  #batch_size, k_value
        topk_index = topk_index_[:,:k_value] #batch_size, k_value

        weights = torch.tensor(topk_prob, dtype=torch.float)
        sample_index = torch.multinomial(weights, 1, replacement=True) #batch_size, 1
        get_index = torch.gather(topk_index, 1, sample_index)
        #batch_size, 1

        ys = torch.cat([ys, get_index], dim=1)
        return ys
