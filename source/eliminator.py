from nltk.tokenize import word_tokenize
from dictionaries.dictionaries import set_negation_indicator
import wllr

class Eliminator(object):
    def __init__(self, detection_info, domain):
        super(Eliminator, self).__init__()
        self.__detection_info = detection_info
        self.__wllr_ranking = wllr.get_wllr_ranking(domain)

    def eliminate(self):
        for info in self.__detection_info:
            info['d_negation'] = [self.__eliminate_sentence(sentence) for sentence in info['d_negation']]
        return self.__detection_info

    def __eliminate_sentence(self, sentence):
        words = word_tokenize(sentence)
        negation_word, eliminated_words = self.__get_eliminated_word(words)
        split_index = words.index(negation_word[0]) - 1
        split_index = 0 if split_index < 0 else split_index
        head = words[:split_index]
        tail = eliminated_words
        new_sentence = ' '.join(head + tail)
        return new_sentence

    def __get_eliminated_word(self, words):
        wllr_ranking = self.__wllr_ranking
        negation_word = list(set(words) & set_negation_indicator)
        words = words[words.index(negation_word[0]) + 1:]

        for i, word in enumerate(words):
            if word in set(wllr_ranking['positive']):
                words[i] = self.__get_opposite_word(word, wllr_ranking['positive'], wllr_ranking['negative'])
            if word in set(wllr_ranking['negative']):
                words[i] = self.__get_opposite_word(word, wllr_ranking['negative'], wllr_ranking['positive'])

        return negation_word, words

    def __get_opposite_word(self, word, source, dest):
        ranking = source.index(word)
        try:
            return dest[ranking]
        except IndexError:
            return dest[-1]
