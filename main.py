import logging
import os

import WordEmbedding

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# the famous reviews data

n_files = 200
index = 0
window = 1

documents = []

file = open('data/perfect-sample.txt', 'r')
text = file.read()


wb = WordEmbedding.WordEmbedding([text], clean_text=True)
print("\n-----------------------------\nVocabulary created\n-----------------------------\n")

matrix = wb.get_faster_term_context_matrix(window=window, separate_sentences=True, show_progress=True)
print("\n-----------------------------\nMatrix created\n-----------------------------\n")

WordEmbedding.WordEmbedding.plot_2d(matrix, wb.vocabulary, f"window size is {window}")