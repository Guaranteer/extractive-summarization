import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import gensim
import nltk
import random

class Batcher(object):

    def __init__(self,params,mode='train'):
        self.params = params
        self.mode = mode

        self.queries = self.loadQueries(self.params['querySource_'+mode], name_len=self.params['name_len_'+mode])
        self.orgDocs, self.feats, self.labels, self.preds = self.loadSents(self.params['pred_csv_'+mode])
        self.keys = list(self.queries.keys())
        self.num = len(self.keys)
        self.count = 0
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(self.params['word2vec'], binary=True)

    def get_data(self):
        query = self.queries[self.keys[self.count]]
        docs = self.orgDocs[self.keys[self.count]]
        if self.mode == 'train':
            label = self.labels[self.keys[self.count]]
        else:
            label = self.preds[self.keys[self.count]]
        self.count += 1
        if self.count == self.num:
            self.count = 0

        scoreList = list()
        length = len(label)
        for i in range(length):
            scoreList.append((label[i], i))
        scoreList.sort(key=lambda x: x[0], reverse=True)

        rewards = list()
        num_pos = 0
        top_docs = list()
        for v, idx in scoreList:
            if num_pos >= 10:
                rewards.append(0)
            else:
                rewards.append(1)
            top_docs.append(docs[idx])
            num_pos += 1
            if num_pos >= 40:
                break
        query_vec, q_len = self.tran_data(query)
        query_vec = query_vec.reshape((1, self.params['s_max_len'], self.params['input_word_dim']))
        q_len = np.array([q_len]).astype(int)

        doc_vec = list()
        d_len = list()
        for sent in top_docs:
            sent_vec, sent_len = self.tran_data(sent)
            doc_vec.append(sent_vec)
            d_len.append(sent_len)
        doc_vec = np.vstack(doc_vec)
        doc_vec = doc_vec.reshape((-1, self.params['s_max_len'], self.params['input_word_dim']))
        d_len = np.array(d_len)

        rewards = np.array(rewards)

        return top_docs, query_vec, q_len, doc_vec, d_len, rewards

    def reset(self):
        self.count = 0
        random.shuffle(self.keys)

    def tran_data(self,sent):
        words = nltk.word_tokenize(sent)
        words = [word.lower() for word in words if word.isalpha()]
        res = np.zeros((self.params['s_max_len'],self.params['input_word_dim']))
        length = 0
        for word in words:
            if word in self.w2v:
                res[length,:] = self.w2v[word]
                length += 1
                if length == self.params['s_max_len']:
                    break
        return res,length

    def loadQueries(self,infName, name_len=5):
        queries = dict()
        with open(infName, 'r') as inf:
            data = inf.read()
            soup = BeautifulSoup(data, 'lxml')

            topics = soup.find_all('topic')
            for topic in topics:
                docid = self._getContent(topic, 'num')[:name_len].lower()
                # title & granularity are omitted
                queries[docid] = self._getContent(topic, 'narr')
        return queries

    def loadSents(self,addr, clean=False):
        all_feat = dict()
        all_sent = dict()
        all_label = dict()
        all_pred = dict()

        res = pd.read_csv(addr)
        data = res.values
        for i in range(len(data)):
            docid = data[i][0]
            length = data[i][8] * 20
            cfeat = [data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7]]
            if clean:
                if length > 8:
                    self._add2DictList(all_feat, docid, cfeat)
                    self._add2DictList(all_sent, docid, data[i][10])
                    self._add2DictList(all_label, docid, data[i][9])
                    self._add2DictList(all_pred, docid, data[i][12])
            else:
                self._add2DictList(all_feat, docid, cfeat)
                self._add2DictList(all_sent, docid, data[i][10])
                self._add2DictList(all_label, docid, data[i][9])
                self._add2DictList(all_pred, docid, data[i][12])

        return all_sent, all_feat, all_label, all_pred

    def _getContent(self,item, name):
        node = item.find(name)
        if node is not None:
            return node.get_text().strip(' \n')
        else:
            return ''

    def _add2DictList(self,dic, k, v):
        if k in dic:
            dic[k].append(v)
        else:
            dic[k] = [v]

    def discount_rewards(self, r, gamma):
        discounted_r = [0.0] * len(r)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add*gamma + r[t]
            discounted_r[t] = running_add * r[t]
        return discounted_r