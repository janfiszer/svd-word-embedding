import logging
import WordEmbedding

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


file = open('data/einstein.txt', 'r')
text = file.read()

window = 1

wb = WordEmbedding.WordEmbedding(text, clean_text=True, min_quantity=10)
matrix = wb.get_term_context_matrix(window=window)

WordEmbedding.WordEmbedding.plot_2d(matrix, wb.vocabulary, f"window size is {window}")
