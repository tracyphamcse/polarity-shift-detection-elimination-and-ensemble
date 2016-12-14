import sys
from configuration import all_domains, list_of_four_classifier
from utils import display_general_help, display_general_domain_not_supported
from document import load_documents_for_train, load_documents_for_test

from common_algorithm import PolarityShiftDetector
from eliminator import Eliminator

from file_io import write_file_by_utf8

def write_detection_result(detection_info, is_train, domain):
    data_dir = 'data_train' if is_train else 'data_test'

    for polarity in set(['positive', 'negative']):
        result_info = [info for info in detection_info if info['polarity'] == polarity]
        print len(result_info)
        for i, info in enumerate(result_info):
            for classifier_name in list_of_four_classifier:
                file_path = '{}/{}/{}/{}/{:03d}.txt'.format(data_dir, domain, polarity, classifier_name, i)
                content = ' '.join(info['d_' + classifier_name])
                write_file_by_utf8(file_path, content)
                

def main(domains, is_train):
    for domain in domains:
        print '----- Domain {} -----'.format(domain)

        print 'Loading documents...'
        documents = load_documents_for_train(domain) if is_train else load_documents_for_test(domain)
        print 'Loaded {} document(s)'.format(len(documents))

        print 'Detecting...'
        detector = PolarityShiftDetector(documents)
        detection_info = detector.detect_for_train() if is_train else detector.detect_for_test()

        print 'Eliminating...'
        eliminator = Eliminator(detection_info, domain)
        detection_info = eliminator.eliminate()

        print 'Generating data train...' if is_train else 'Generating data test...'
        write_detection_result(detection_info, is_train, domain)

def help():
    print 'Usage: python polarity_shift_detector.py [-<option>] [<domain>]'
    print '\tpython polarity_shift_detector.py -train all\t: to run train all domains\n'
    print '\tpython polarity_shift_detector.py -test all\t: to run test all domains\n'
    print 'Supported options: train, test'
    print 'Supported domains: ' + ', '.join(all_domains) + '\n'
    print 'If no argument, print help'

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        print 'Arguments not found!'
        help()
        sys.exit()

    domains = all_domains

    if 'all' not in set_args:
        domains = list(set_args & set(all_domains))

    if not domains:
        display_general_domain_not_supported()
        sys.exit()

    if '-train' not in set_args and '-test' not in set_args:
        print 'Option not found!'
        help();
        sys.exit()

    if '-train' in set_args:
        main(domains, True)

    if '-test' in set_args:
        main(domains, False)

    print 'Done'
