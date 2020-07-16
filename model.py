import csv

from gensim.models import nmf, LdaModel

from dataset_create import create_dataset
import utils.nlp_functions as nlp


'''
Script que gera modelo baseado em um dataset (em csv)
Modelos:
    NMF
    LDA
'''

# Se vai precisar gerar dataset
PRE_PROC = False

# data path
PATH_DATA = 'data_raw'
FOLDER = 'data_pi'

# dataset
SET_NAME = 'dataset_data_pi'

# train path
PATH_TRAIN = 'trained_models'
MODEL_NAME = 'NMF_PI'

# Model constants
MODEL_TYPE = 'NMF'
QT_TOPICS = 15


if PRE_PROC:
    path_dir = f'{PATH_DATA}/{FOLDER}'
    texts = create_dataset(path_dir)

    print('Foram pre processados {0} documentos'.format(len(texts)))

else:
    # abre dataset ja preprocessado
    with open(f'dataset/{SET_NAME}.csv', 'r') as csv_file:
        texts = csv.reader(csv_file, delimiter=',')
        texts = list(texts)
        texts = [text for text in texts if text]

# Cria Matriz termo-documento (tr_idf) e dicionario
corpus, dictionary = nlp.prepare_corpus_dictinary(texts)
corpus_tfidf = nlp.tfidf_matrix(corpus)

tam_ds = sum(len(x) for x in texts)
print('Dataset tem {0} palavras unicas e \n'
      ' {1} palavras totais'.format(len(dictionary.token2id), tam_ds))


if MODEL_TYPE == 'LDA':
    # https://radimrehurek.com/gensim/models/ldamodel.html
    result_model = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=QT_TOPICS, update_every=0, passes=15,
                            alpha='symmetric', eta=1.0, decay=0.9, offset=1.0, eval_every=1000, iterations=1000,
                            gamma_threshold=0.1, minimum_probability=0.05)

    print(result_model.print_topics(num_topics=QT_TOPICS))

    result_model.save(f'{PATH_TRAIN}/{MODEL_TYPE.lower()}/{MODEL_NAME}')

elif MODEL_TYPE == 'NMF':
    # https://radimrehurek.com/gensim/models/nmf.html
    result_model = nmf.Nmf(corpus_tfidf, id2word=dictionary, num_topics=QT_TOPICS, passes=20, chunksize=2000,
                           kappa=1.0, normalize=True, minimum_probability=0.01, w_max_iter=200, w_stop_condition=0.0001,
                           h_max_iter=50, h_stop_condition=0.001, eval_every=10, random_state=1000)

    print(result_model.print_topics(num_topics=QT_TOPICS))

    result_model.save(f'{PATH_TRAIN}/{MODEL_TYPE.lower()}/{MODEL_NAME}')
