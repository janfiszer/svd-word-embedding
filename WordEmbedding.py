import numpy as np
import re
import logging
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from typing import List


class WordEmbedding:
    """
    Class dedicated to perform SVD word embedding on gives set of texts
    @:param texts- list of texts on which the embedding will be perform
    @:param min_quantity- an integer representing minimum number of a word in whole text set to be in vocabulary
    @:param clean_text- bool
    """
    def __init__(self, texts: List[str], min_quantity=1, clean_text=False, include_stop_words=True):
        self.texts = texts
        # booleans
        self.include_stop_words = include_stop_words
        self.clean_text = clean_text

        self.min_quantity = min_quantity

        self.vocabulary = self.get_vocabulary()
        self.full_vocabulary = self._get_full_vocabulary()

    @staticmethod
    def text_to_words(texts: List[str], discard_dots=True) -> List[str]:
        if discard_dots:
            regex_exp = '[^A-Za-z]+'
        else:
            # not sure bout this regex
            regex_exp = '[^A-Za-z.]+'

        all_text = ""

        for text in texts:
            text = re.sub(regex_exp, ' ', text)
            all_text += text

        lower_text = all_text.lower()

        return lower_text.split()

    def _get_full_vocabulary(self) -> np.array:
        words = self.text_to_words(self.texts)

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

        vocabulary.sort()

        return np.array(vocabulary)

    # function to make term-context matrix
    # first I was using vocabulary instead of full_vocabulary, but then word embedding lose much info
    # so trying other aproach
    def get_term_context_matrix(self, window=1, separate_sentences=True, debug=False):
        words = self.text_to_words(self.texts, discard_dots=not separate_sentences)

        n = len(self.full_vocabulary)
        matrix = np.zeros((n, n))

        # do it less complicated way
        for i in range(len(words)):
            # print(words[i])
            print(f"{i/len(words)}")
            for j in range(1, window + 1):
                # TODO: make it less C
                has_dot = False
                word = words[i]
                x = np.where(self.full_vocabulary == word)

                if word[-1] == '.':
                    word = word.replace('.', '')
                    has_dot = True
                if debug:
                    logging.debug(f"i = {i}")

                # IF FACT IT DOESN'T NEED TO CHECK PREVIOUS
                # if i != 0:
                #     prev_word = words[i - j]
                #     if not prev_word[-1] == '.':
                #         if debug:
                #             logging.debug(f"Pointed word:{word}")
                #             logging.debug(f"{j} word behind:{prev_word}")
                #
                #         y1 = np.where(self.full_vocabulary == prev_word)
                #
                #         # when V is provided then not every word have to be in there
                #         if debug:
                #             logging.debug(x, y1)
                #         matrix[x, y1] += 1/j
                #         # matrix[y1, x] += 1/j
                if not has_dot:
                    if i + j < len(words):
                        next_word = words[i + j].replace('.', '')
                        if debug:
                            logging.debug(f"Pointed word:{word}")
                            logging.debug(f"{j} word forward:{next_word}")

                        y2 = np.where(self.full_vocabulary == next_word)
                        # when V is provided then not every word have to be in there
                        if debug:
                            logging.debug(x, y2)
                        matrix[x, y2] += 1/j
                        matrix[y2, x] += 1/j


                if debug:
                    logging.debug("")

        matrix = self._full_matrix_reduce(matrix)
        # since function _full_matrix_reduce() modifies
        self.full_vocabulary = self._get_full_vocabulary()

        return matrix

    def get_words_quantities(self) -> dict:
        if self.clean_text:
            words = self.text_to_words(self.texts)
        else:
            words = []
            for text in self.texts:
                lower_text = text.lower()
                words.append(lower_text.split())

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
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

        text_column = []

        for key, value in words_quantities.items():
            text_column.append(f"{key}: {value}")

        df = pd.DataFrame([U[:, 0], U[:, 1], text_column])
        df = df.transpose()

        fig = px.scatter(df, x=0, y=1, text=2)
        fig.update_traces(textposition='top center')
        fig.show()
