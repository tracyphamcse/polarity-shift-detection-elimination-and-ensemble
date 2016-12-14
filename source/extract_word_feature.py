from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import svm

import numpy as np
import pickle

import sys
import time

from file_io import read_file, write_file
from utils import all_domains
from nltk.tokenize import word_tokenize





def init_data(d_type,domain, d_name):
    NUMBER_OF_FILE = 500
    positive = [read_file('{}/{}/{}/{}/{:03d}.txt'.format(d_type, domain, 'positive', d_name, i)) for i in range(NUMBER_OF_FILE)]
    negative = [read_file('{}/{}/{}/{}/{:03d}.txt'.format(d_type, domain, 'negative', d_name, i)) for i in range(NUMBER_OF_FILE)]

    positive = positive[:500]
    negative = negative[:500]

    positive = [p for p in positive if p]
    negative = [n for n in negative if n]

    print 'Number of positive: ', len(positive)
    print 'Number of negative: ', len(negative)

    data = positive + negative
    return data, len(positive), len(negative)

def generate_label(n_pos, n_neg):
    return [1] * n_pos + [0] * n_neg

def extract_features(d_type,domain, d_name):
    print 'Extracting feature for {} in domain {}, set {}'.format(d_type, domain, d_name)
    data,_,_ = init_data(d_type,domain,d_name)
    N_GRAM = [1,1]	#Unigram
    # N_GRAM = [1,2]	#Bigram
    vectorizer = TfidfVectorizer(decode_error='ignore',binary=True, ngram_range=N_GRAM)
    # vectorizer = TfidfVectorizer(decode_error='replace', ngram_range = (1,2))
    X_train_counts = vectorizer.fit_transform(data)

    with open('resources/{}/{}/pickles/vectorizer.{}.pk'.format(d_type,domain, d_name), 'wb') as fin:
        pickle.dump(vectorizer, fin)
    print 'Store at {}/{}/pickles/vectorizer.{}.pk'.format(d_type,domain, d_name) + '\n'

def main(d_type, domains):
    for domain in domains:
        extract_features(d_type, domain, 'contrast')
        extract_features(d_type, domain, 'negation')
        extract_features(d_type, domain, 'inconsistence')
        extract_features(d_type, domain, 'no_shift')
        extract_features(d_type, domain, 'based')
        print '----'

def help():
    print 'Usage: python extract_word_feature.py [d_type [<domain>]]'
    print '\tpython training.py [d_type all]\t to run all domains in d_type\n'
    print 'Supported d_type: data_train, data_test (default: data_train)\n'
    print 'Supported domains: ' + ', '.join(all_domains) + '\n'
    print 'If no argument, print help'

def domain_not_supported():
    print 'Domain not found'
    print 'Supported domains: ' + ', '.join(all_domains)

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        help()
        sys.exit()

    domains = all_domains

    if 'all' not in set_args:
        domains = list(set_args & set(all_domains))

    if not domains:
        domain_not_supported()
        sys.exit()

    d_type = ''
    if 'test' in set_args:
        d_type = 'data_test'
    else:
        d_type = 'data_train'

    main(d_type, domains)
