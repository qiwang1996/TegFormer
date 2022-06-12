fo = open('temp_1.txt', 'r+', encoding='utf-8')
topic2extension = json.load(fo)
fo.close()

fo = open('temp_2.txt', 'r+', encoding='utf-8')
kw2freq = json.load(fo)
fo.close()


used_kw = {}
topic2kws = {}
for topic, extension in topic2extension.items():
    extension = [[kw, score_freq] for kw, score_freq in extension.items()]
    extension = sorted(extension, key=lambda x: x[1][1], reverse=True)[:20]
    freq_cnt = sum([score_freq[1] for kw, score_freq in extension])
    extension = [kw for kw, score_freq in extension
                 if kw in kw2freq.keys() and
                 score_freq[1] / kw2freq[kw] > 0.01 and
                 score_freq[1] / freq_cnt > 0.01 and
                 kw2freq[kw] > 8]
    for kw in extension:
        if used_kw.get(kw, None) is None:
            used_kw[kw] = 0
        used_kw[kw] += 1
    topic2kws[topic] = extension

fo = open('used_kw.txt', 'w+', encoding='utf-8')
used_kw = [[kw, cnt] for kw, cnt in used_kw.items()]
used_kw = sorted(used_kw, key=lambda x: x[1], reverse=True)
used_kw = [[kw, cnt] for [kw, cnt] in used_kw if cnt > 4]
for kw, cnt in used_kw:
    fo.write('{}\t{}\n'.format(kw, cnt))

used_kw = {kw: cnt for kw, cnt in used_kw}

fo = open('res.txt', 'w+', encoding='utf-8')
for topic, kws in topic2kws.items():
    kws = [kw for kw in kws if kw in used_kw.keys()]
    fo.write('{}\t{}\n'.format(topic, '\t'.join(kws)))














