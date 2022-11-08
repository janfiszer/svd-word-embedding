import logging
import WordEmbedding

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


file = open('einstein.txt', 'r')
text = file.read()

wb = WordEmbedding.WordEmbedding(text)

V = wb.get_vocabulary(min_quantity=10, clean_text=True, include_stop_words=False)
matrix = wb.get_term_context_matrix(vocabulary=V, window=10)
WordEmbedding.WordEmbedding.plot_2d(matrix, V)
