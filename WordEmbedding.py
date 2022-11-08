import numpy as np
import re
import logging
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords


# TODO: make class Vocabulary ?
# TODO: do not ignore dots
# TODO: improve function plot_2d_with_occurrences_number()
# TODO: deal with " in given file

class WordEmbedding:

    def __init__(self, text: str, min_quantity=1, clean_text=False, include_stop_words=True):
        self.text = text
        # booleans
        self.include_stop_words = include_stop_words
        self.clean_text = clean_text

        self.min_quantity = min_quantity

        self.vocabulary = self.get_vocabulary()
        self.full_vocabulary = self._get_full_vocabulary()

    def _text_to_words(self, text: str, discard_dots=True) -> list[str]:
        if discard_dots:
            regex_exp = '[^A-Za-z]+'
        else:
            regex_exp = '[^A-Za-z.]+'

        text = re.sub(regex_exp, ' ', text)
        lower_text = text.lower()
        words = lower_text.split()
        return words

    def _get_full_vocabulary(self) -> np.array:
        words = self._text_to_words(self.text)
        return np.unique(words)

    def _get_index_in_vocabulary(self, word) -> int:
        index = np.where(self.full_vocabulary == word)
        if index:
            return index[0]
        else:
            return -1

    def _full_matrix_reduce(self, matrix):
        words_to_drop = []

        for word in self.full_vocabulary:
            if word not in self.vocabulary:
                words_to_drop.append(word)

        for word in words_to_drop:
            index = self._get_index_in_vocabulary(word)
            matrix = np.delete(matrix, index, 0)
            matrix = np.delete(matrix, index, 1)

            # we have to remove also from full_vocabulary to don't lose track where are the words
            self.full_vocabulary = np.delete(self.full_vocabulary, index)

        return matrix

    def get_vocabulary(self):
        words = self.get_words_quantities()

        vocabulary = []

        for word, quantity in words.items():
            if quantity >= self.min_quantity:
                vocabulary.append(word)

        if not self.include_stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            vocabulary = [w for w in vocabulary if w not in stop_words]

        return np.array(vocabulary)

    # function to make term-context matrix
    # first I was using vocabulary instead of full_vocabulary, but then word embedding lose much info
    # so trying other aproach
    def get_term_context_matrix(self, window=1, discard_dots=True, debug=False):
        words = self._text_to_words(self.text, discard_dots=discard_dots)

        n = len(self.full_vocabulary)
        matrix = np.zeros((n, n))

        # do it less complicated way
        for i in range(len(words)):
            print(words[i])
            for j in range(1, window + 1):
                # TODO: make it less C
                has_dot = False
                word = words[i]
                x = np.where(self.full_vocabulary == word)

                if word[-1] != '.':
                    word = word.replace('.', '')
                    has_dot = True
                if debug:
                    logging.debug(f"i = {i}")

                if i != 0:
                    if debug:
                        logging.debug(f"Pointed word:{word}")
                        logging.debug(f"{j} word behind:{words[i - j]}")

                    y1 = np.where(self.full_vocabulary == words[i - j])

                    # when V is provided then not every word have to be in there
                    if debug:
                        logging.debug(x, y1)
                    matrix[x, y1] += 1/j

                if not has_dot:
                    if i + j < len(words):
                        if debug:
                            logging.debug(f"Pointed word:{word}")
                            logging.debug(f"{j} word forward:{words[i + j]}")

                        y2 = np.where(self.full_vocabulary == words[i + j])
                        # when V is provided then not every word have to be in there
                        if debug:
                            logging.debug(x, y2)
                        matrix[x, y2] += 1/j

                if debug:
                    logging.debug("")

        matrix = self._full_matrix_reduce(matrix)
        # since function _full_matrix_reduce() modifies
        self.full_vocabulary = self._get_full_vocabulary()

        return matrix

    def get_words_quantities(self) -> dict:
        if self.clean_text:
            words = self._text_to_words(self.text)
        else:
            lower_text = self.text.lower()
            words = lower_text.split()

        V = {}

        for word in words:
            if word not in V:
                V[word] = 1
            else:
                V[word] += 1

        if self.min_quantity > 1:
            # TODO: make one for loop, not sure if I can do it anyway
            words_to_delete = []
            for word, value in V.items():
                if self.min_quantity > value:
                    words_to_delete.append(word)

            for word in words_to_delete:
                del V[word]

        return V

    @staticmethod
    def plot_2d(matrix, vocabulary: list[str], title: str):
        U, s, Vh = np.linalg.svd(matrix)

        df = pd.DataFrame([U[:, 0], U[:, 1], vocabulary])
        df = df.transpose()

        fig = px.scatter(df, x=0, y=1, text=2, title=title)
        fig.update_traces(textposition='top center')
        fig.show()

    # NOT REALLY USEFUL
    @staticmethod
    def plot_2d_with_occurrences_number(matrix, words_quantities: dict[str, int]):
        U, s, Vh = np.linalg.svd(matrix)

        text_column = []

        for key, value in words_quantities.items():
            text_column.append(f"{key}: {value}")

        df = pd.DataFrame([U[:, 0], U[:, 1], text_column])
        df = df.transpose()

        fig = px.scatter(df, x=0, y=1, text=2)
        fig.update_traces(textposition='top center')
        fig.show()
