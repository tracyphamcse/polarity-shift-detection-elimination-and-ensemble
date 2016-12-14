from nltk.corpus import stopwords
from file_io import read_file
from handle_unprocessed_data import normalize_word

def init_stop_words():
    stop_words = set(stopwords.words('english'))
    extra_stop_words = read_file('dictionaries/stopwords.txt').split('\n')
    extra_stop_words = [unicode(word, 'utf-8') for word in extra_stop_words if word]
    stop_words.update(extra_stop_words)
    return set([normalize_word(word) for word in stop_words])

def init_set_in_file(file_path):
    words = read_file(file_path).split('\n')
    return set([word for word in words if word])

set_stop_words = init_stop_words()
set_accepted_tags = init_set_in_file('dictionaries/accepted_tags.txt')
set_negation_indicator = init_set_in_file('dictionaries/negation_indicator.txt')
set_fore_contrast_indicator = init_set_in_file('dictionaries/fore_contrast_indicator.txt')
set_post_contrast_indicator = init_set_in_file('dictionaries/post_contrast_indicator.txt')
set_clause_indicator = init_set_in_file('dictionaries/clause_indicator.txt')
