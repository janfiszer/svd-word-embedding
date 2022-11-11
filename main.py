import logging
import WordEmbedding
import os
import pickle

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# n_files = 200
# index = 0
#
# texts = []
#
# for root, dirs, files in os.walk('data/reviews/neg'):
#     for filename in files:
#         if index < n_files:
#             file = open(os.path.join(root, filename))
#             try:
#                 text = file.read()
#                 texts.append(text)
#             except UnicodeDecodeError:
#                 print(f"File: {filename} wasn't loaded successfully.")
#             index += 1


file = open('data/perfect-sample.txt', 'r')
text = file.read()

window = 1

wb = WordEmbedding.WordEmbedding([text], clean_text=True)
print("\n-----------------------------\nVocabulary created\n-----------------------------\n")
print(len(wb.full_vocabulary), wb.vocabulary.shape)
matrix = wb.get_term_context_matrix(window=window, separate_sentences=True)
print("\n-----------------------------\nMatrix created\n-----------------------------\n")
# print(matrix.shape)
# f = open('matrix-reviews-seconds-try.pkl', 'wb')
# pickle.dump(matrix, f)

# matrix = pickle.load(open('matrix.pkl', 'rb'))
# print(matrix.shape)

WordEmbedding.WordEmbedding.plot_2d(matrix, wb.full_vocabulary, f"window size is {window}")