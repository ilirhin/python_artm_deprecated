import numpy as np
from numpy.core.umath_tests import inner1d
import scipy
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
import gensim
from collections import Counter
from collections import defaultdict
import heapq
import nltk
import random
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time


def prepare_sklearn_dataset(dataset, calc_cooccurences=False, train_test_split=None, token_2_num=None):
    english_stopwords = set(stopwords.words('english'))
    is_token_2_num_provided = token_2_num is not None 
    # remove stopwords
    if not is_token_2_num_provided:
        token_2_num = {}
        occurences = Counter()
        for i, doc in enumerate(dataset.data):
            tokens = gensim.utils.lemmatize(doc)
            for token in set(tokens):
                occurences[token] += 1
            if i % 500 == 0:
                print 'Processed: ', i, 'documents from', len(dataset.data)
    
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    doc_targets = []
    doc_cooccurences = Counter()
    doc_occurences = Counter()
    random_gen = random.Random(42)
    
    for doc, target in zip(dataset.data, dataset.target):
        tokens = gensim.utils.lemmatize(doc)
        cnt = Counter()
        cnt_test = Counter()
        for token in tokens:
            word = token.split('/')[0]
            if not is_token_2_num_provided and word not in english_stopwords and 3 <= occurences[token] and token not in token_2_num:
                token_2_num[token] = len(token_2_num)
            if token in token_2_num:
                if train_test_split is None or random_gen.random() < train_test_split:
                    cnt[token_2_num[token]] += 1
                else:
                    cnt_test[token_2_num[token]] += 1
        
        if len(cnt) > 0 and (train_test_split is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)
                
            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)
                
            not_empty_docs_number += 1
            doc_targets.append(target)
            
            if calc_cooccurences:
                words = set(cnt.keys() + cnt_test.keys())
                doc_occurences.update(words)
                doc_cooccurences.update({(w1, w2) for w1 in words for w2 in words if w1 != w2})
        
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }
    print 'Nonzero values:', len(data)
    shape = (len(doc_targets), len(token_2_num))
    if train_test_split is None:
        if calc_cooccurences:
            return scipy.sparse.csr_matrix((data, (row, col)), shape=shape), token_2_num, num_2_token, doc_targets, doc_occurences, doc_cooccurences
        else:
            return scipy.sparse.csr_matrix((data, (row, col)), shape=shape), token_2_num, num_2_token, doc_targets
    else:
        if calc_cooccurences:
            return (
                scipy.sparse.csr_matrix((data, (row, col)), shape=shape),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test)), shape=shape),
                token_2_num,
                num_2_token,
                doc_targets,
                doc_occurences,
                doc_cooccurences
            )
        else:
            return (
                scipy.sparse.csr_matrix((data, (row, col))),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test))),
                token_2_num,
                num_2_token,
                doc_targets
            )
        
        
def prepare_nips_dataset(dataset_path, calc_cooccurences=False, train_test_split=None):
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    doc_targets = []
    doc_cooccurences = Counter()
    doc_occurences = Counter()
    random_gen = random.Random(42)
    token_2_num = {}
    documents = defaultdict(list)

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f.xreadlines()):
            if i % 1000 == 0:
                print 'Read file lines:', i
            if i > 0:
                tokens = line.strip().split(',')
                token_2_num[tokens[0][1:-1]] = i - 1
                for doc_num, val in enumerate(tokens[1:]):
                    value = int(val)
                    if value > 0:
                        documents[doc_num].append((i - 1, value))
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }

    for doc_num, words in documents.iteritems():
        if doc_num % 100 == 0:
            print 'Processed documents:', doc_num
        
        cnt = Counter()
        cnt_test = Counter()
        
        for word_num, number in words:
            for _ in xrange(number):
                if train_test_split is None or random_gen.random() < train_test_split:
                    cnt[word_num] += 1
                else:
                    cnt_test[word_num] += 1

        if len(cnt) > 0 and (train_test_split is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)
                
            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)
                
            not_empty_docs_number += 1
            
            if calc_cooccurences:
                keys = [x for x, _ in words]
                doc_cooccurences.update({(w1, w2) for w1 in keys for w2 in keys if w1 != w2})
                doc_occurences.update(keys)

    print 'Nonzero values:', len(data)
    shape = (len(doc_targets), len(token_2_num))
    
    if train_test_split is None:
        if calc_cooccurences:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token, doc_occurences, doc_cooccurences
        else:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token
    else:
        if calc_cooccurences:
            return (
                scipy.sparse.csr_matrix((data, (row, col)), shape=shape),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test)), shape=shape),
                token_2_num,
                num_2_token,
                doc_occurences,
                doc_cooccurences
            )
        else:
            return (
                scipy.sparse.csr_matrix((data, (row, col)), shape=shape),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test)), shape=shape),
                token_2_num,
                num_2_token
            )