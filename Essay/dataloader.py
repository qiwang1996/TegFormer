from transformers import BertModel, BertTokenizer
from transformer_model import subsequent_mask
import torch
import random

class TextTokenizer:
    def __init__(self, bert_path, pad_id, pad_size):
        self.bert = BertModel.from_pretrained(bert_path, output_hidden_states=True, from_tf=False)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.pad_id = pad_id
        self.pad_size = pad_size


    def token2id(self, s):
        pad_id = self.pad_id
        pad_size = self.pad_size
        tokens = self.tokenizer.tokenize('[CLS]' + s + '[SEP]')
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pad_token_ids = (token_ids + [pad_id] * pad_size)[:pad_size]
        token_mask = ([1] * len(token_ids) + [0] * pad_size)[:pad_size]
        tokens_tensor = torch.tensor(pad_token_ids)
        mask_tensor = torch.tensor(token_mask)
        return [tokens_tensor, mask_tensor]

class TopicTokenizer:
    def __init__(self, topic_file, pad_id, pad_size):
        self.topic_to_id = {}
        self.id_to_topic = {}
        self.pad_id = pad_id
        self.pad_size = pad_size
        id = 0
        with open(topic_file, 'r') as tfile:
            for line in tfile:
                topic = line.strip()
                self.topic_to_id[topic] = id
                self.id_to_topic[id] = topic
                id += 1

    def token2id(self, s):
        topic_id = []
        topic_mask = []
        for topic in s:
            if topic in self.topic_to_id.keys():
                topic_id.append(self.topic_to_id[topic])
                topic_mask.append(1)
            else:
                topic_id.append(self.pad_id)
                topic_mask.append(0)

        topic_id = topic_id + [self.pad_id] * self.pad_size
        topic_id = topic_id[:self.pad_size]
        topic_id = torch.LongTensor(topic_id)

        topic_mask = topic_mask + [0] * self.pad_size
        topic_mask = topic_mask[:self.pad_size]
        topic_mask = torch.tensor(topic_mask)

        return topic_id, topic_mask

class Dataloader:
    def __init__(self, data_file, topic_file, bert_path, text_pad_id, text_pad_size,
                 topic_pad_id, topic_pad_size, batch_size, is_train=False):

        self.is_train = is_train
        lines = open(data_file, 'r+').readlines() 
        if is_train:
            text_ids = []
            text_masks = []
            tokenizer_text = TextTokenizer(bert_path, text_pad_id, text_pad_size)

        topic_ids = []
        topic_masks = []
        tokenizer_topic = TopicTokenizer(topic_file, topic_pad_id, topic_pad_size)
        self.data_len = len(lines)
        for line in lines:
            if is_train:
                text, topic = line.strip().split('</d>')
                token_id, token_mask = tokenizer_text.token2id(text)
                text_ids.append(token_id)
                text_masks.append(token_mask)
            else:
                topic = line.strip()

            topic = topic.split()
            token_id, token_mask = tokenizer_topic.token2id(topic)
            topic_ids.append(token_id)
            topic_masks.append(token_mask)

        if is_train:
            self.text_ids = torch.stack(text_ids, dim=0)
            self.text_masks = torch.stack(text_masks, dim=0)
        self.topic_ids = torch.stack(topic_ids, dim=0)
        self.topic_masks = torch.stack(topic_masks, dim=0)


        self.shuffle()
        self.batch_size = batch_size
        self.batch_num = self.data_len // self.batch_size

    def shuffle(self):
        if self.is_train:
            shuffle_list = [i for i in range(self.data_len)]
            random.shuffle(shuffle_list)
            self.text_ids = self.text_ids[shuffle_list]
            self.text_masks = self.text_masks[shuffle_list]
            self.topic_ids = self.topic_ids[shuffle_list]
        self.g_point = 0

    def get_batch(self):
        if self.g_point == self.batch_num and self.is_train:
            self.shuffle()
        self.g_point += 1

        if self.is_train:
            batch_text_id = self.text_ids[(self.g_point - 1) * self.batch_size: self.g_point * self.batch_size]
            batch_text_mask = self.text_masks[(self.g_point - 1) * self.batch_size: self.g_point * self.batch_size]

        batch_topic_id = self.topic_ids[(self.g_point - 1) * self.batch_size: self.g_point * self.batch_size]
        batch_topic_mask = self.topic_masks[(self.g_point - 1) * self.batch_size: self.g_point * self.batch_size]

        if self.is_train:
            return Batch(batch_topic_id, batch_topic_mask, batch_text_id, batch_text_mask)
        else:
            return Batch(batch_topic_id, batch_topic_mask)

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, src_mask, trg=None, trg_mask=None,  pad=0):
        self.src = src
        self.src_mask = src_mask
        if trg is not None:
            trg_mask = trg_mask[:, :-1]
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(trg_mask)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(trg_mask):
        "Create a mask to hide padding and future words."
        size = trg_mask.size(-1)
        tgt_mask = trg_mask.unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(size).type_as(tgt_mask.data)
        return tgt_mask














