from __future__ import division

from nltk.tokenize import sent_tokenize, word_tokenize
from dictionaries.dictionaries import set_negation_indicator, set_fore_contrast_indicator, set_post_contrast_indicator, set_stop_words, set_accepted_tags, set_clause_indicator
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class PolarityShiftDetector(object):
    def __init__(self, documents):
        super(PolarityShiftDetector, self).__init__()
        self.__documents = documents
        self.__detection_info = []
        import wllr
        self.__wllr_value = wllr.get_wllr_in_domain(documents[0].get_domain())

    def detect_for_train(self):
        self.__is_training = True
        return self.__run_detection()

    def detect_for_test(self):
        self.__is_training = False
        return self.__run_detection()

    def __run_detection(self):
        for i, document in enumerate(self.__documents):
            # print 'Detecting document {} ...'.format(i)
            self.__detect_polarity_shift(document)
        return self.__detection_info

    def __detect_polarity_shift(self, document):
        d_negation, d_contrast, d_inconsistence, d_no_shift = [], [], [], []
        sentences = document.get_sentences()
        polarity = document.get_polarity() if self.__is_training else self.__get_predict_polarity(sentences, document.get_domain())

        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            if self.__is_contain_negation_word(words):
                d_negation.append(sentence)
                continue

            if (self.__is_contain_post_contrast_word(words)) or \
                (i + 1 < len(sentences) and self.__is_contain_fore_contrast_word(word_tokenize(sentences[i + 1]))):
                d_contrast.append(sentence)
                continue

            h_of_sentence = self.__compute_h(words, polarity)
            if h_of_sentence < 0:
                d_inconsistence.append(sentence)

        d_no_shift = list(set(sentences) - set(d_negation) - set(d_contrast) - set(d_inconsistence))

        self.__detection_info.append({
            # 'content': '\n'.join(document.get_sentences()),
            'polarity': document.get_polarity(),
            'd_negation': d_negation,
            'd_contrast': d_contrast,
            'd_inconsistence': d_inconsistence,
            'd_no_shift': d_no_shift
        })

    def __is_contain_negation_word(self, words):
        return len(set_negation_indicator & set(words))

    def __is_contain_fore_contrast_word(self, words):
        return len(set_fore_contrast_indicator & set(words))

    def __is_contain_post_contrast_word(self, words):
        return len(set_post_contrast_indicator & set(words))

    def __find_intersection_word(self, set1, set2):
        return list(set1 & set2)[0] or ''

    def __compute_h(self, words, polarity):
        y = self.__compute_y(polarity)
        f = self.__compute_f(words)
        return y * f

    def __compute_y(self, polarity):
        return 1 if polarity == 'positive' else -1

    def __compute_f(self, words):
        r_of_words = [self.__get_r_of_word(word) for word in words]
        return sum(r_of_words)

    def __get_r_of_word(self, word):
        try:
            return self.__wllr_value[word]
        except KeyError as e:
            return 0

    def __get_predict_polarity(self, sentences, domain):
        sentences = [word_tokenize(sentence) for sentence in sentences]
        r_of_words = [self.__get_r_of_word(word) for sentence in sentences for word in sentence]

        if sum(r_of_words) < 0:
            return 'negative'
        return 'positive'

    # def __get_predict_polarity(self, sentences, domain):
    #     tfidf = pickle.load(open('resources/data_train/{}/pickles/vectorizer.processed.pk'.format(domain), 'rb'))
    #     classifier = pickle.load(open('resources/data_train/{}/pickles/classifier.processed.pk'.format(domain), 'rb'))
    #
    #     test_vectors = tfidf.transform([' '.join(sentences)])
    #     predict = classifier.predict(test_vectors)
    #
    #     if (predict[0] == 1):
    #         return 'positive'
    #     else:
    #         return 'negative'


# configure Stanford POS Tagger
from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
import platform

stanford_pos_dir = 'resources/libs/stanford-postagger-2015-12-09/'
eng_model_filename = stanford_pos_dir + 'models/english-left3words-distsim.tagger'
my_path_to_jar = stanford_pos_dir + 'stanford-postagger.jar'

tagger = StanfordPOSTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar)
# https://gist.github.com/alvations/e1df0ba227e542955a8a
# http://stackoverflow.com/questions/34361725/nltk-stanfordnertagger-noclassdeffounderror-org-slf4j-loggerfactory-in-windo
stanford_jars = find_jars_within_path(stanford_pos_dir)
separator = ';' if 'Windows' in platform.platform() else ':'
tagger._stanford_jar = separator.join(stanford_jars)
# End configuration

class WLLR(object):
    def __init__(self, documents):
        super(WLLR, self).__init__()
        self.__documents = documents
        self.__set_contrast_indicator = set_fore_contrast_indicator | set_post_contrast_indicator

        print 'Initialize document info...'
        document_info = self.__init_document_info(documents)

        print 'Initialize dictionary...'
        self.__dictionary = {}
        self.__init_dictionary(document_info)

        print 'Calculating WLLR...'
        self.__calculate_WLLR_all_words()

    def __init_document_info(self, documents):
        return filter(lambda info: info['words'], [self.__get_document_info(document, i) for i, document in enumerate(documents)])

    def __get_document_info(self, document, index):
        print 'Get info in document {}/{}'.format(index + 1, len(self.__documents))
        sentences = document.get_sentences()
        sentences = filter(lambda sent: self.__is_not_negation_or_contrast(sent), sentences)
        list_of_words = [self.__get_accepted_words(sentence) for sentence in sentences]

        words = [word for words in list_of_words for word in words]
        return {
            'words': words,
            'polarity': document.get_polarity()
        }

    def __is_not_negation_or_contrast(self, sentence):
        words = word_tokenize(sentence)
        return not(self.__is_contain_negation_word(words) or self.__is_contain_contrast_word(words))

    def __get_accepted_words(self, sentence):
        list_of_word__tag = tagger.tag(word_tokenize(sentence))
        return [word__tag[0] for word__tag in list_of_word__tag if word__tag[1] in set_accepted_tags]

    def __is_contain_negation_word(self, words):
        return len(set_negation_indicator & set(words))

    def __is_contain_contrast_word(self, words):
        return len(self.__set_contrast_indicator & set(words))

    def __init_dictionary(self, document_info):
        for info in document_info:
            words = [word for word in info['words'] if self.__is_valid_word(word)]
            self.__add_all_words_to_dictionary(words, info['polarity'])

    def __is_valid_word(self, word):
        return word not in set_stop_words \
            and len(word) > 2 \
            and any(character.isalpha() for character in word)

    def __add_all_words_to_dictionary(self, words, polarity):
        for word in words:
            self.__add_word_to_dictionary(word, polarity)

    def __add_word_to_dictionary(self, word, polarity):
        if word in self.__dictionary:
            self.__dictionary[word]['appearance_number_in_' + polarity] += 1
            return

        info = self.__init_word_info()
        info['appearance_number_in_' + polarity] = 1
        self.__dictionary[word] = info

    def __init_word_info(self):
        info = {}
        info['appearance_number_in_positive'] = 1
        info['appearance_number_in_negative'] = 1
        return info

    def __calculate_WLLR_all_words(self):
        dictionary = self.__dictionary
        for word in dictionary:
            if (dictionary[word]['appearance_number_in_positive'] + dictionary[word]['appearance_number_in_negative']) <= 5:
                dictionary[word]['r'] = 0
                self.__dictionary[word]['r_positive'] = 0
                self.__dictionary[word]['r_negative'] = 0
                continue

            p_of_word_given_positive = dictionary[word]['appearance_number_in_positive'] / 1000
            p_of_word_given_negative = dictionary[word]['appearance_number_in_negative'] / 1000
            r_positive = p_of_word_given_positive * math.log10(p_of_word_given_positive / p_of_word_given_negative)
            r_negative = p_of_word_given_negative * math.log10(p_of_word_given_negative / p_of_word_given_positive)
            dictionary[word]['r'] = self.__compute_r(r_positive, r_negative)

            self.__dictionary[word]['r_positive'] = r_positive
            self.__dictionary[word]['r_negative'] = r_negative

    def __compute_r(self, r_positive, r_negative):
        r = r_positive - r_negative
        return r if abs(r) >= 0.001 else 0

    def get_r_of_word(self, word):
        try:
            return self.dictionary[word]['r']
        except KeyError as e:
            return 0

    def get_dictionary(self):
        return self.__dictionary


from handle_unprocessed_data import normalize_sentence

class ClauseSplitter():
    def __init__(self, sentence):
        super(ClauseSplitter, self).__init__()

    @staticmethod
    def split_sentence(sentence):
        sentences = ClauseSplitter.split_sentence_by_clause_indicator(sentence)

        sentences = [ClauseSplitter.split_sentence_for_negation(sentence) for sentence in sentences]
        sentences = [sent for sentence in sentences for sent in sentence]

        sentences = [ClauseSplitter.split_sentence_for_contrast(sentence) for sentence in sentences]
        sentences = [sent for sentence in sentences for sent in sentence]

        return sentences

    @staticmethod
    def split_sentence_for_negation(sentence):
        clauses = []
        sentences = sentence.split(';')
        for sentence in sentences:
            subsentences = ClauseSplitter.split_sentence_by_negation_indicator(sentence)
            clauses.extend(subsentences)

        clauses = [normalize_sentence(clause) for clause in clauses]
        return clauses

    @staticmethod
    def split_sentence_by_negation_indicator(sentence):
        negation_word = ClauseSplitter.find_negation_word(sentence)

        if not negation_word:
            return [sentence]

        sentences = []

        index_of_negation_word = sentence.find(negation_word)
        index_of_comma = sentence.find(',', 15, index_of_negation_word)
        if index_of_comma > -1:
            head, tail = sentence[:index_of_comma], sentence[index_of_comma + 1:]
            sentences.append(head)
            sentence = tail
            index_of_negation_word = sentence.find(negation_word)

        index = []
        index.append(index_of_negation_word)
        i = 0
        while sentence.find(',', index[i] + 1) > -1:
            new_index = sentence.find(',', index[i] + 1)
            if new_index == index[i]:
                break
            index.append(new_index)
            i += 1
        index.remove(index_of_negation_word)

        split_index = -1
        for i in range(len(index)):
            begin = index[i]
            end = len(sentence) if i == len(index) - 1 else index[i + 1]
            phrase = sentence[begin:end]
            words = word_tokenize(phrase)
            if len(words) > 5:
                split_index = begin
                break
        if split_index == -1:
            sentences.append(sentence)
            return sentences

        first_clause, second_clause = sentence[:split_index], sentence[split_index + 1:]
        sentences.extend([first_clause, second_clause])
        return sentences

    @staticmethod
    def split_sentence_for_contrast(sentence):
        clauses = []
        sentences = sentence.split(';')
        for sentence in sentences:
            subsentences = ClauseSplitter.split_sentence_by_contrast_indicator(sentence)
            clauses.extend(subsentences)

        clauses = [normalize_sentence(clause) for clause in clauses]
        return clauses

    @staticmethod
    def split_sentence_by_contrast_indicator(sentence):
        contrast_word = ClauseSplitter.find_contrast_word(sentence)
        if not contrast_word:
            return [sentence]

        if ClauseSplitter.is_begining_of_sentence(contrast_word, sentence):
            return ClauseSplitter.split_sentence_have_contrast_word_at_first(sentence)

        return ClauseSplitter.split_sentence_have_contrast_word_at_middle(sentence, contrast_word)

    @staticmethod
    def split_sentence_by_clause_indicator(sentence):
        set_words = set(word_tokenize(sentence))
        clause_indicator = ClauseSplitter.find_intersection_word(set_words, set_clause_indicator)

        if not clause_indicator:
            return [sentence]

        if ClauseSplitter.is_begining_of_sentence(clause_indicator, sentence):
            return [sentence]

        indicator_index = sentence.index(clause_indicator)
        clauses = [sentence[:indicator_index], sentence[indicator_index:]]
        clauses = [normalize_sentence(clause) for clause in clauses]
        return clauses

    @staticmethod
    def is_begining_of_sentence(word, sentence):
        words = word_tokenize(sentence)
        if words.index(word) in [0,1,2]:
            return True
        return False

    @staticmethod
    def split_sentence_have_contrast_word_at_first(sentence):
        phrases = sentence.split(',')
        if len(phrases) == 2:
            return [' '.join(phrases)]

        middle_phrase_index = int(math.ceil(len(phrases) / 2))
        first_clause, second_clause = phrases[:middle_phrase_index], phrases[middle_phrase_index:]

        begin_index_of_second_clause = -1
        for i, phrase in enumerate(second_clause):
            words = word_tokenize(phrase)
            if len(words) < 5:
                first_clause.append(phrase)
            else:
                begin_index_of_second_clause = i
                break

        first_clause = ' '.join(first_clause)
        if begin_index_of_second_clause == -1:
            return [first_clause]

        second_clause = ' '.join(second_clause[begin_index_of_second_clause:])
        return [first_clause, second_clause]

    @staticmethod
    def split_sentence_have_contrast_word_at_middle(sentence, contrast_word):
        split_index = sentence.index(contrast_word)
        first_clause, second_clause = sentence[:split_index], sentence[split_index:]
        return [first_clause, second_clause]

    @staticmethod
    def find_negation_word(clause):
        set_words = set(word_tokenize(clause))
        return ClauseSplitter.find_intersection_word(set_words, set_negation_indicator)

    @staticmethod
    def find_contrast_word(clause):
        set_words = set(word_tokenize(clause))
        return ClauseSplitter.find_intersection_word(set_words, set_fore_contrast_indicator) or \
                ClauseSplitter.find_intersection_word(set_words, set_post_contrast_indicator)

    @staticmethod
    def find_intersection_word(set1, set2):
        words = list(set1 & set2)
        return words[0] if words else ''

if __name__ == '__main__':
    # print ClauseSplitter.split_sentence_for_contrast('Although I like apple, orange, I do not like banana'.lower())
    # print ClauseSplitter.split_sentence_for_negation('I don\'t like apple, orange and I like banana while I don\'t like orange'.lower())

    sentences = []
    string = 'what you want from a supervisor ,  but it is not very accurate.'.lower()
    temp_sentences = sent_tokenize(string)
    for sentence in temp_sentences:
        sentences.extend(ClauseSplitter.split_sentence(sentence))
    # print sentences
    # print len(sentences)
