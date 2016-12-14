from file_io import load_files_in_directory
from nltk.tokenize import sent_tokenize, word_tokenize

class Document(object):
    def __init__(self, content, index, domain, polarity):
        super(Document, self).__init__()
        self.__content = content
        self.__sentences = sent_tokenize(content)
        self.__index = index
        self.__domain = domain
        self.__polarity = polarity

    def get_content(self):
        return self.__content

    def get_sentences(self):
        return self.__sentences

    def add_sentence(self, sentence):
        self.__sentences.append(sentence)

    def remove_sentence(self, sentence):
        self.__sentences.remove(sentence)

    def get_index(self):
        return self.__index

    def get_domain(self):
        return self.__domain

    def get_polarity(self):
        return self.__polarity

    def set_polarity(self, polarity):
        self.__polarity = polarity

    def __get_r_of_word(self, word):
        try:
            return self.wllr_value[word]
        except KeyError as e:
            return 0

def load_documents_for_train(domain):
    positive_data = load_files_in_directory('data_train/{}/{}/processed/'.format(domain, 'positive'))
    negative_data = load_files_in_directory('data_train/{}/{}/processed/'.format(domain, 'negative'))

    positive_documents = [Document(data, i, domain, 'positive') for i, data in enumerate(positive_data)]
    negative_documents = [Document(data, i, domain, 'negative') for i, data in enumerate(negative_data)]

    return positive_documents + negative_documents

def load_documents_for_test(domain):
    positive_data = load_files_in_directory('data_test/{}/{}/processed/'.format(domain, 'positive'))
    negative_data = load_files_in_directory('data_test/{}/{}/processed/'.format(domain, 'negative'))

    positive_documents = [Document(data, i, domain, 'positive') for i, data in enumerate(positive_data)]
    negative_documents = [Document(data, i, domain, 'negative') for i, data in enumerate(negative_data)]

    return positive_documents + negative_documents
