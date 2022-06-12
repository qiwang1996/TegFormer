from jieba import analyse
from tqdm import tqdm
import json


# 引入TextRank关键词抽取接口
text_rank = analyse.textrank
# 原始文本

stopwords = [stopw.strip() for stopw in open('stopwords.txt', encoding='utf-8')]
topic_set = [topic.strip() for topic in open('topic_dict.txt', encoding='utf-8')]
topic2extension = {topic:{} for topic in topic_set}
kw2freq = {}

word2cnt = {}

train_cnt = 50000
for line in tqdm(open('zhihu.txt', encoding='utf-8')):
    if train_cnt == 0:
        break
    train_cnt -= 1
    text, topics = line.split('</d>')
    text = text.strip()
    topics = topics.strip().split()
    kw2score = [[kw, text_rank_score] for kw, text_rank_score in
                          text_rank(text, withWeight=True, topK=8) if kw not in stopwords + topics]
    for topic in topics:
        for kw, score in kw2score:
            if topic2extension[topic].get(kw) is None:
                topic2extension[topic][kw] = [0, 0]
            topic2extension[topic][kw][0] += score
            topic2extension[topic][kw][1] += 1

    for kw in text.split():
        if kw not in kw2freq.keys():
            kw2freq[kw] = 0
        kw2freq[kw] += 1

fsave = open('temp_1.txt', 'w+', encoding='utf-8')
json.dump(topic2extension, fsave, ensure_ascii=False, indent=2)
fsave = open('temp_2.txt', 'w+', encoding='utf-8')
json.dump(kw2freq, fsave, ensure_ascii=False, indent=2)
exit()
