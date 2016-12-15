from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import numpy as np
import pickle

import sys
import time

from file_io import read_file_by_utf8, write_file
from utils import all_domains
from nltk.tokenize import word_tokenize

from extract_word_feature import init_data, generate_label


def train(domain, d_name):

    print 'Domain {}, set {}'.format(domain, d_name)

    data_train,n_train_pos, n_train_neg = init_data('data_train',domain, d_name)

    label_train = generate_label(n_train_pos, n_train_neg)

    tfidf = pickle.load(open('resources/data_train/{}/pickles/vectorizer.{}.pk'.format(domain, d_name), 'rb'))
    tfidf = tfidf.transform(data_train)

    # print 'Preparing for cross validation...'
    # train_vectors, test_vectors, train_labels, test_labels = \
    #     cross_validation.train_test_split(tfidf, label_train, test_size=0.3, random_state=43)

    train_vectors = tfidf
    train_labels = label_train

    # print 'Initializing SVM model...'
    # C_range = 10.0 ** np.arange(-4, 4)
    # gamma_range = 10.0 ** np.arange(-4, 4)
    # degree_range = np.arange(1,5)
    # param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
    # svr = svm.SVC(kernel='linear',degree = 2, probability = True)
    # classifier = GridSearchCV(svr, param_grid)

    # classifier = SVC(random_state=1234, kernel='linear', probability=True)				#SVM MODEL
    classifier = LogisticRegression(random_state=1234)

    print 'Training...'
    t0 = time.time()

    classifier.fit(train_vectors, train_labels)
    t1 = time.time()
    # print 'Testing...'
    # prediction = classifier.predict(test_vectors)
    # t2 = time.time()

    time_train = t1 - t0
    # time_predict = t2 - t1

    # report = 'The best classifier for polynominal is:\n%s\n' % classifier.best_estimator_
    # report += 'Results for SVC(kernel=linear)\n'
    # report += 'Training time: %fs; Prediction time: %fs\n' % (time_train, time_predict)
    # report += classification_report(test_labels, prediction)
    # report += '\nThe accuracy score is {:.2%}'.format(accuracy_score(test_labels, prediction))
    # print '\n----\nResult\n\----\n' + report
    # print '\n\n\n'
    # print '--------------------------------------'
    print 'Writing model...'
    with open('resources/data_train/{}/pickles/classifier.{}.pk'.format(domain, d_name), 'wb') as f:
        pickle.dump(classifier, f)

    # return classifier, report

def test(d_type,domain, d_name):
    NUMBER_OF_FILE = 500

    positive = [read_file_by_utf8('{}/{}/{}/{}/{:03d}.txt'.format(d_type, domain, 'positive', d_name, i)) for i in range(NUMBER_OF_FILE)]
    negative = [read_file_by_utf8('{}/{}/{}/{}/{:03d}.txt'.format(d_type, domain, 'negative', d_name, i)) for i in range(NUMBER_OF_FILE)]

    tfidf = pickle.load(open('resources/data_train/{}/pickles/vectorizer.{}.pk'.format(domain, d_name), 'rb'))
    classifier = pickle.load(open('resources/data_train/{}/pickles/classifier.{}.pk'.format(domain, d_name), 'rb'))

    ground_labels = []
    predict_labels = []
    feature_labels = []


    for p in positive:
        if (p):
            test_vectors = tfidf.transform([p])

            ground_labels.append(1)

            predict = classifier.predict(test_vectors)
            predict_labels.extend(predict)

            feature = classifier.predict_proba(test_vectors)
            feature_labels.extend(feature)

        else:
            feature_labels.append([1,1])

    for n in negative:
        if (n):
            test_vectors = tfidf.transform([n])

            ground_labels.append(0)

            predict = classifier.predict(test_vectors)
            predict_labels.extend(predict)

            feature = classifier.predict_proba(test_vectors)
            feature_labels.extend(feature)

        else:
            feature_labels.append([1,1])

    print 'data', len(positive), len(negative)
    print 'ground', set(ground_labels), len(ground_labels)
    print 'prec', set(predict_labels), len(predict_labels)

    report = 'Test on {}/{}/{}'.format(d_type, domain, d_name)
    report += classification_report(ground_labels, predict_labels)
    report += '\nThe accuracy score is {:.2%}'.format(accuracy_score(ground_labels, predict_labels))
    print '\n----\nResult\n----\n' + report
    print '\n\n\n'
    print '--------------------------------------'

    with open('resources/{}/{}/pickles/ensemble_feature.{}.pk'.format(d_type, domain, d_name), 'wb') as f:
        pickle.dump(feature_labels, f)


def main(domains):
    for domain in domains:
        train(domain, 'negation')
        # # write_file('data_train/' + domain + '/eliminated-report.txt', report)
        #
        train(domain, 'contrast')
        # # write_file('data_train/' + domain + '/contrast-report.txt', report)
        #
        train(domain, 'inconsistence')
        # # write_file('data_train/' + domain + '/inconsistence-report.txt', report)
        #
        train(domain, 'no_shift')
        # # write_file('data_train/' + domain + '/no_shift-report.txt', report)

        train(domain, 'processed')
        # write_file('data_train/' + domain + '/no_shift-report.txt', report)

def main_test(domains):
    for domain in domains:
        # test('data_train',domain, 'negation')
        test('data_test',domain, 'negation')

        # test('data_train',domain, 'contrast')
        test('data_test',domain, 'contrast')

        # test('data_train',domain, 'inconsistence')
        test('data_test',domain, 'inconsistence')

        # test('data_train',domain, 'no_shift')
        test('data_test',domain, 'no_shift')

        # test('data_train',domain, 'processed')
        test('data_test',domain, 'processed')


def help():
    print 'Usage: python training.py [<domain>]'
    print '\tpython training.py all\t to run all domains\n'
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

    if 'test' in set_args:
        main_test(domains)
    else:
        main(domains)
