import logging
import os

import WordEmbedding

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# the famous reviews data

n_files = 200
index = 0
window = 2

documents = []

for root, dirs, files in os.walk('data/reviews/neg'):
    for filename in files:
        if index < n_files:
            file = open(os.path.join(root, filename))
            try:
                text = file.read()
                documents.append(text)
            except UnicodeDecodeError:
                print(f"File: {filename} wasn't loaded successfully.")
            index += 1


wb = WordEmbedding.WordEmbedding(documents, clean_text=True, min_quantity=10, include_stop_words=False)
print("\n-----------------------------\nVocabulary created\n-----------------------------\n")

matrix = wb.get_faster_term_context_matrix(window=window, separate_sentences=True, show_progress=True)
print("\n-----------------------------\nMatrix created\n-----------------------------\n")

WordEmbedding.WordEmbedding.plot_2d(matrix, wb.vocabulary, f"window size is {window}")