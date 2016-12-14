import sys
from configuration import all_domains
from utils import display_general_help, display_general_domain_not_supported

from file_io import read_file, write_file_by_utf8
from document import load_documents_for_train, load_documents_for_test

from common_algorithm import WLLR

def get_wllr_in_domain(domain):
    wllr = {}
    wllr.update(get_wllr_in_domain_and_polarity(domain, 'positive'))
    wllr.update(get_wllr_in_domain_and_polarity(domain, 'negative'))
    return wllr

def get_wllr_in_domain_and_polarity(domain, polarity):
    wllr = {}
    file_path = 'WLLR/{}.{}.wllr'.format(domain, polarity)
    lines = read_file(file_path).split('\n')
    lines.remove('')
    for line in lines:
        word, wllr_index = line.split('\t')
        wllr[word] = float(wllr_index)
    return wllr

def get_wllr_ranking(domain):
    ranking = {}

    file_path = 'WLLR/{}.positive.wllr'.format(domain)
    lines = read_file(file_path).split('\n')
    lines.remove('')
    ranking['positive'] = [line.split('\t')[0] for line in lines]

    file_path = 'WLLR/{}.negative.wllr'.format(domain)
    lines = read_file(file_path).split('\n')
    lines.remove('')
    ranking['negative'] = [line.split('\t')[0] for line in lines]

    return ranking

def generate_report(list_words):
    report_content = ''
    for word_info in list_words:
        line = word_info[0] + '\t' + str(word_info[1]) + '\n'
        report_content += line
    return report_content

def main(domains, is_train):
    for domain in domains:
        print 'Reading documents...'
        documents = load_documents_for_train(domain) if is_train else load_documents_for_test(domain)
        print 'Read {} documents in domain {}'.format(len(documents), domain)

        wllr = WLLR(documents)

        print 'Writing WLLR...'
        dictionary = wllr.get_dictionary()

        list_positive = [(word, word_info['r']) for word, word_info in dictionary.iteritems() if word_info['r'] > 0]
        list_negative = [(word, word_info['r']) for word, word_info in dictionary.iteritems() if word_info['r'] < 0]

        list_positive = sorted(list_positive, key=lambda w: w[1], reverse=True)
        report_content = generate_report(list_positive)
        file_path = 'WLLR/train/' + domain + '.positive.wllr' if is_train else  'WLLR/test/' + domain + '.positive.wllr'
        write_file_by_utf8(file_path, report_content)

        list_negative = sorted(list_negative, key=lambda w: w[1], reverse=False)
        report_content = generate_report(list_negative)
        file_path = 'WLLR/train/' + domain + '.negative.wllr' if is_train else  'WLLR/test/' + domain + '.negative.wllr'
        write_file_by_utf8(file_path, report_content)

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='wllr')
        sys.exit()

    domains = all_domains

    if 'all' not in set_args:
        domains = list(set_args & set(all_domains))

    if not domains:
        display_general_domain_not_supported()
        sys.exit()

    if '-train' not in set_args and '-test' not in set_args:
        print 'Option not found!'
        sys.exit()

    if '-train' in set_args:
        main(domains, True)

    if '-test' in set_args:
        main(domains, False)
    print 'Done!'
