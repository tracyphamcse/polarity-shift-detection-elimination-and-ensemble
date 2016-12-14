import sys
from configuration import all_domains
from utils import display_general_help, display_general_domain_not_supported, StringUtils
from file_io import read_file_by_utf8, write_file_by_utf8

from bs4 import BeautifulSoup as Soup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def parse_to_review_contents(review_data):
    soup = Soup(review_data, 'lxml')
    reviews = soup.find_all('review_text')
    review_contents = [review.contents[0] for review in reviews]
    return review_contents

def normalize_content(document):
    sentences = sent_tokenize(document)
    sentences = [normalize_sentence(sentence) for sentence in sentences if StringUtils.is_not_empty(sentence)]
    return ' '.join(sentences)

def normalize_sentence(sentence):
    sentence = sentence.lower().replace('/',' ')
    words = word_tokenize(sentence)
    words = [normalize_word(word) for word in words if StringUtils.is_not_empty(word)]
    sentence = ' '.join(words)
    sentence = add_dot_at_end_of_line(sentence.strip(' \t\n\r'))
    return sentence

def normalize_word(word):
    # return lemmatizer.lemmatize(word)
    return word

def add_dot_at_end_of_line(sentence):
    if len(sentence) == 0:
        return sentence
    if sentence[-1] in ['.', '?', '!', ',']:
        sentence = sentence[:len(sentence) - 1]
    return sentence.strip(' \t\n\r') + '.'

def handle_unprocessed_data(domain, polarity):
    review_data = read_file_by_utf8('unprocessed/{}/{}.review'.format(domain, polarity))
    review_contents = parse_to_review_contents(review_data)
    review_contents = [split_sentence(review_content) for review_content in review_contents]

    data_for_train = [review_content for i, review_content in enumerate(review_contents) if i % 2 is not 0]
    data_for_test = [review_content for i, review_content in enumerate(review_contents) if i % 2 is 0]

    for i, data in enumerate(data_for_train):
        write_file_by_utf8('data_train/{}/{}/{}/{:03d}.txt'.format(domain, polarity, 'processed', i), data)
    for i, data in enumerate(data_for_test):
        write_file_by_utf8('data_test/{}/{}/{}/{:03d}.txt'.format(domain, polarity, 'processed', i), data)

    return len(data_for_train) + len(data_for_test)

def split_sentence(review_content):
    from common_algorithm import ClauseSplitter
    sentences = []
    temp_sentences = sent_tokenize(review_content)
    for sentence in temp_sentences:
        sentences.extend(ClauseSplitter.split_sentence(sentence))
    return ' '.join(sentences)

def main(domains):
    for domain in domains:
        print 'Handling domain {}, polarity {}...'.format(domain, 'positive')
        number_of_handled_files = handle_unprocessed_data(domain, 'positive')
        print 'Handled {} file(s)'.format(number_of_handled_files)

        print 'Handling domain {}, polarity {}...'.format(domain, 'negative')
        number_of_handled_files = handle_unprocessed_data(domain, 'negative')
        print 'Handled {} file(s)'.format(number_of_handled_files)

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='handle_unprocessed_data')
        sys.exit()

    domains = all_domains

    if 'all' not in set_args:
        domains = list(set_args & set(all_domains))

    if not domains:
        display_general_domain_not_supported()
        sys.exit()

    main(domains)
    print 'Done!'
