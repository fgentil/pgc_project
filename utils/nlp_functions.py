import re

from gensim import models, corpora
from collections import defaultdict

from nltk.corpus import stopwords
stopwords = stopwords.words('portuguese')

stopnames = open('utils/stopnames.txt', encoding='utf-8').read().splitlines()
stopwords = stopwords + stopnames

import spacy

pln = spacy.load('pt_core_news_lg')

import unidecode


def prepoc_tokenizer(text):
    '''
    Funçao para pre processamento de texto, retira stopwords e lematiza as
    palavras recebidas de um texto
    :param text: str
    '''

    ts = pln(text)
    # pega todos as palavras, em caixa baixa classificadas como substantivos que nao sejam stopwords
    words = [txt.lemma_.lower() for txt in ts if txt.pos_ == 'NOUN']

    # retira numeros e pontuacao
    words = [word for word in words if word.isalpha()]

    # remove links e emails
    words = [word for word in words if not re.search('www|http|\.com|º|@', word)]

    # remove acentos
    words = [unidecode.unidecode(word) for word in words if len(word) > 2
             and unidecode.unidecode(word) not in stopwords]

    return words


def prepare_corpus_dictinary(texts):
    '''
    Funçao que gera um corpus e um dicionario a partir de um texto
    :param texts - lista de lista de string
    '''
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    # remove palavras que aparecem apenas uma única vez
    texts_clean = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    # prepara o corpus e a lista de tokens (dicionário)
    dictionary = corpora.Dictionary(texts_clean)
    corpus = [dictionary.doc2bow(txt) for txt in texts_clean]

    return corpus, dictionary


def tfidf_matrix(corpus):
    '''
    Funçao que gera uma matrix termo documento, recebe um corpus
    :param corpus: lista de lista de tuplas
    '''
    tfidf = models.TfidfModel(corpus)  # inicializa o modelo TF-IDF
    corpus_tfidf = tfidf[corpus]  # cria a matriz termo-documento

    return corpus_tfidf

