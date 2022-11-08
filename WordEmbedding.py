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

    def __init__(self, text: str, discard_dots=True):
        self.text = text
        self.full_vocabulary = self._get_full_vocabulary()
        self.discard_dots = discard_dots

    @staticmethod
    def _text_to_words(text: str, discard_dots=True) -> list[str]:
        if discard_dots:
            regex_exp = '[^A-Za-z]+'
        else:
            regex_exp = '[^A-Za-z.]+'

        text = re.sub(regex_exp, ' ', text)
        lower_text = text.lower()
        words = lower_text.split()
        return words

    @staticmethod
    def plot_2d(matrix, vocabulary: list[str]):
        U, s, Vh = np.linalg.svd(matrix)

        df = pd.DataFrame([U[:, 0], U[:, 1], vocabulary])
        df = df.transpose()

        fig = px.scatter(df, x=0, y=1, text=2)
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

    def _get_full_vocabulary(self) -> np.array:
        words = self._text_to_words(self.text)
        return np.unique(words)

    def get_vocabulary(self, min_quantity, clean_text=False, include_stop_words=True, discard_dots=True):
        words = self.get_words_quantities(clean_text=clean_text)

        vocabulary = []

        for word, quantity in words.items():
            if quantity >= min_quantity:
                vocabulary.append(word)

        if not include_stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            vocabulary = [w for w in vocabulary if w not in stop_words]

        return np.array(vocabulary)

    # function to make term-context matrix
    # TODO: make it better when the window is > 1
    def get_term_context_matrix(self, vocabulary=None, window=1, debug=False):
        words = self._text_to_words(self.text)

        if vocabulary is None:
            vocabulary = self.full_vocabulary

        n = len(vocabulary)
        matrix = np.zeros((n, n))

        # do it less complicated way
        for i in range(len(words)):
            print(words[i])
            for j in range(1, window + 1):
                # TODO: make it less C
                x = np.where(vocabulary == words[i])

                if debug:
                    logging.debug(f"i = {i}")

                if i != 0:
                    if debug:
                        logging.debug(f"Pointed word:{words[i]}")
                        logging.debug(f"{j} word behind:{words[i - j]}")

                    y1 = np.where(vocabulary == words[i - j])

                    # when V is provaided then not every word have to be in there
                    #                 if np.any(y2) != 0:
                    if debug:
                        logging.debug(x, y1)
                    matrix[x, y1] += 1/j

                if i + j < len(words):
                    if debug:
                        logging.debug(f"Pointed word:{words[i]}")
                        logging.debug(f"{j} word forward:{words[i + j]}")
                    # try:
                    #     y2 = np.where(vocabulary == words[i + j])
                    # except IndexError:
                    #     logging.warning(f"i = {i}  j = {j}")

                    y2 = np.where(vocabulary == words[i + j])
                    # when V is provided then not every word have to be in there
                    if debug:
                        logging.debug(x, y2)
                    matrix[x, y2] += 1/j

                if debug:
                    logging.debug("")

        return matrix

    def get_words_quantities(self, clean_text=False, min_quantity=1, discard_dots=True) -> dict:
        if clean_text:
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

        if min_quantity > 1:
            # TODO: make one for loop, not sure if I can do it anyway
            words_to_delete = []
            for word, value in V.items():
                if min_quantity > value:
                    words_to_delete.append(word)

            for word in words_to_delete:
                del V[word]

        return V
