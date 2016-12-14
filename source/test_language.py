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

from extract_word_feature import init_data, generate_label



if __name__ == '__main__':

    domains = all_domains
    d_name  = ['eliminateds', 'contrasts', 'inconsistences', 'no-shifts']
    for domain in domains:
        for name in d_name:
            classifier = pickle.load(open('data_train/{}/pickles/classifier.{}.pk'.format(domain,name), 'rb'))
            data_train = init_data('data_test',domain, name)
            vectorizer = pickle.load(open('data_train/{}/pickles/vectorizer.{}.pk'.format(domain, name), 'rb'))
            test_vectors = vectorizer.transform(data_train)

            prediction = classifier.predict(test_vectors)

            y = [1]*500 + [0]*500

            report = 'Report of {}/{}'.format(domain, name)
            report += classification_report(y, prediction)
            report += '\nThe accuracy score is {:.2%}'.format(accuracy_score(y, prediction))
            print '\n----\nResult\n\----\n' + report
            print '\n\n\n'
            print '--------------------------------------'
