import logging
import WordEmbedding
import os

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


# f1 = open('data/physicists/einstein.txt', 'r')
# t1 = f1.read()
# f2 = open('data/physicists/planck.txt', 'r')
# t2 = f2.read()
# f3 = open('data/physicists/bohr.txt', 'r')
# t3 = f3.read()
# f4 = open('data/physicists/heisenberg.txt', 'r')
# t4 = f4.read()
texts = []

for root, dirs, files in os.walk('data/reviews/neg'):
    for filename in files:
        file = open(os.path.join(root, filename))
        try:
            text = file.read()
            texts.append(text)
        except UnicodeDecodeError:
            print(f"File: {filename} wasn't loaded successfully.")


window = 2

wb = WordEmbedding.WordEmbedding(texts, clean_text=True, min_quantity=10, include_stop_words=False)
print("\n-----------------------------\nVocabulary created\n-----------------------------\n")
matrix = wb.get_term_context_matrix(window=window, separate_sentences=True)
print("\n-----------------------------\nMatrix created\n-----------------------------\n")
WordEmbedding.WordEmbedding.plot_2d(matrix, wb.vocabulary, f"window size is {window}")


