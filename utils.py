import pkuseg
import math
import random
from tqdm import tqdm,trange
import codecs

label_map = {'财经': 0, '教育': 1, '房产': 2, '娱乐': 3, '游戏': 4,
             '体育': 5, '时尚': 6, '科技': 7, '时政': 8, '家居': 9}
def read_corpus(file_path):
    """读取语料
    :param file_path:
    :param type:
    :return:
    """
    src_data = []
    labels = []
    seg = pkuseg.pkuseg()
    with codecs.open(file_path,'r',encoding='utf-8') as fout:
        for line in tqdm(fout.readlines(),desc='reading corpus'):
            if line is not None:
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    print(pair)
                    continue
                src_data.append(seg.cut(pair[1]))
                labels.append(pair[0])
    return (src_data, labels)

def pad_sents(sents,pad_token):
    """pad句子"""
    sents_padded = []
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    for sent in sents:
        sent_padded = sent + [pad_token] * (max_len - len(sent))
        sents_padded.append(sent_padded)
    return sents_padded

def batch_iter(data, batch_size, shuffle=False):
    """
        batch数据
    :param data: list of tuple
    :param batch_size:
    :param shuffle:
    :return:
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        random.shuffle(index_array)

    for i in trange(batch_num,desc='get mini_batch data'):
        indices = index_array[i*batch_size:(i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples,key=lambda x: len(x[1]),reverse=True)
        src_sents = [e[0] for e in examples]
        labels = [label_map[e[1]] for e in examples]

        yield src_sents, labels
