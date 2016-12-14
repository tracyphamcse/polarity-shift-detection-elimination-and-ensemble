from extract_word_feature import init_data, generate_label
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import cross_validation

import numpy as np
import pickle

import sys
import time
from utils import all_domains
from file_io import read_file, write_file


def flat_label_pred(label):
    result = []
    for l in label:
        result.extend(l)
    return result


def get_feature(d_type, domain):

    dump_feature = {}
    for d_name in ['contrast', 'inconsistence', 'negation', 'no_shift', 'based']:
        dump_feature[d_name] = pickle.load(open('resources/{}/{}/pickles/ensemble_feature.{}.pk'.format(d_type, domain, d_name), 'rb'))
    # print dump_feature['contrast']

    feature = zip(dump_feature['contrast'], dump_feature['inconsistence'], dump_feature['negation'], dump_feature['no_shift'], dump_feature['no_shift'],dump_feature['no_shift'], dump_feature['based'], dump_feature['based'], dump_feature['based'])
    # feature = zip(dump_feature['contrast'], dump_feature['inconsistence'], dump_feature['negation'], dump_feature['no_shift'], dump_feature['no_shift'], dump_feature['no_shift'])

    feature = [flat_label_pred(f) for f in feature]

    return feature

def train(domain):

    print 'Getting feature...'

    X = get_feature('data_train', domain)
    y = [1] * 500 + [0]*500

    print 'Training....'

    train_vectors, test_vectors, train_labels, test_labels = \
        cross_validation.train_test_split(X, y, test_size=0.3, random_state=43)
    classifier = LogisticRegression(random_state=1234)
    t0 = time.time()
    classifier.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction = classifier.predict(test_vectors)
    t2 = time.time()

    time_train = t1 - t0
    time_predict = t2 - t1

    report = 'Results on Validation set\n'
    report += 'Training time: %fs; Prediction time: %fs\n' % (time_train, time_predict)
    report += classification_report(test_labels, prediction)
    report += '\nThe accuracy score is {:.2%}'.format(accuracy_score(test_labels, prediction))
    print '\n------\nResult\n------\n' + report
    print '\n\n\n'

    with open('resources/data_train/{}/pickles/classifier.ensemble.pk'.format(domain), 'wb') as f:
        pickle.dump(classifier, f)
    return classifier, report

def test_product_rule(data, target):
    y_pred = []
    for d in data:
        if (d[0] * d[2] * d[4] * d[6] * d[8] * d[10] * d[12] * d[14] * d[16] > d[1] * d[3] * d[5] * d[7] * d[9] * d[11] * d[13] * d[15] * d[17] ):
            y_pred.append(0)
        else:
            y_pred.append(1)

    acc = accuracy_score(target, y_pred)
    report = classification_report(target, y_pred)

    print 'Accuracy: ' , acc
    print report

def test(domain, classifier):
    X = get_feature('data_test', domain)
    y = [1] * 500 + [0]*500

    prediction = classifier.predict(X)

    # test_product_rule(X, y);


    report = 'Result on Test set\n'
    report += classification_report(y, prediction)
    report += '\nThe accuracy score is {:.2%}'.format(accuracy_score(y, prediction))
    print '\n------\nResult\n------\n' + report
    print '\n\n\n'
    return report

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

    set_args = set(sys.argv[1:])

    domains = all_domains

    if 'all' not in set_args:
        domains = list(set_args & set(all_domains))

    for domain in domains:
        print 'Training LogisticRegression on {} domain \n'.format(domain)
        classifier, report = train(domain)
        write_file('data_train/' + domain + '/ensemble-report.txt', report)
        report = test(domain, classifier)
        write_file('data_test/' + domain + '/ensemble-report.txt', report)
