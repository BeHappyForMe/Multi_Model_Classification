from utils import read_corpus,pad_sents

from typing import List
from collections import Counter
from itertools import chain
import json
import torch

class VocabEntry(object):
    def __init__(self,word2id=None):
        """
        初始化vocabEntry
        :param word2id: mapping word to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0
            self.word2id['<UNK>'] = 1
        self.unk_id = self.word2id['<UNK>']
        self.id2word = {v:k for k,v in self.word2id.items()}

    def __getitem__(self,word):
        """获取word的idx"""
        return self.word2id.get(word,self.unk_id)

    def __contains__(self,word):
        return word in self.word2id

    def __setitem__(self,key,value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)
    def __repr__(self):

        return 'Vocabulary[size=%d]' % (len(self.word2id))

    def add(self,word):
        """增加word"""
        if word not in self.word2id:
            wid = self.word2id[word] = len(self.word2id)
            self.id2word[wid] = word
            return wid
        else:
            return self.word2id[word]

    def words2indices(self,sents):
        """
        将sents转为number index
        :param sents: list(word) or list(list(wod))
        :return:
        """
        if type(sents[0]) == list:
            return [[self.word2id.get(w,self.unk_id) for w in s] for s in sents]
        else:
            return [self.word2id.get(s,self.unk_id) for s in sents]

    def indices2words(self,idxs):
        return [self.id2word[id] for id in idxs]

    def to_input_tensor(self,sents: List[List[str]], device: torch.device):
        """
        将原始句子list转为tensor,同时将句子PAD成max_len
        :param sents: list of list<str>
        :param device:
        :return:
        """
        sents = self.words2indices(sents)
        sents = pad_sents(sents,self.word2id['<PAD>'])
        sents_var = torch.tensor(sents,device=device)
        return sents_var

    @staticmethod
    def from_corpus(corpus,size,min_feq = 3):
        """从给定语料中创建VocabEntry"""
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = word_freq.most_common(size-2)
        valid_words = [word for word, value in valid_words if value >= min_feq]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), min_feq, len(valid_words)))
        for word in valid_words:
            vocab_entry.add(word)
        return vocab_entry

class Vocab(object):
    """src、tgt的词汇类"""
    def __init__(self, src_vocab: VocabEntry, labels: dict):
        self.vocab = src_vocab
        self.labels = labels

    @staticmethod
    def build(src_sents,labels, vocab_size, min_feq):

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents,vocab_size,min_feq)

        return Vocab(src,labels)

    def save(self,file_path):
        with open(file_path,'w') as fint:
            json.dump(dict(src_word2id=self.vocab.word2id,labels=self.labels),fint,indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path,'r') as fout:
            entry = json.load(fout)
        src_word2id = entry['src_word2id']
        labels = entry['labels']

        return Vocab(VocabEntry(src_word2id),labels)
    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words)' % (len(self.vocab))


if __name__ == '__main__':

    src_sents,labels = read_corpus('/Users/zhoup/develop/NLPSpace/my-whole-data/cnews/cnews.train.txt')
    labels = {label:idx for idx,label in enumerate(labels)}

    vocab = Vocab.build(src_sents,labels, 50000, 3)
    print('generated vocabulary, source %d words' % (len(vocab.vocab)))
    vocab.save('./vocab.json')
