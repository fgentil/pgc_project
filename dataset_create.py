import os
import csv

from gensim.models import Phrases
import utils.nlp_functions as nlp


def get_files_list(data_path):
    '''
    Funçao que busca arquivos em pasta retorna lista de arquivos
    :param path_dir: str    - local onde se encontra dados nao preprocessados
    :return files: list     - lista de arquivos
    '''
    files = os.listdir(data_path)

    return files


def create_dataset(path_dir):
    '''
    Funçao para gerar dataset preprocessado em csv
    :param path_dir: str    - local onde se encontra dados nao preprocessados
    :return texts: list     - lista de listas de str
    '''
    # path_dir = PATH_DATA + FOLDER
    files = get_files_list(path_dir)
    # Pre Processamento dos textos
    texts = []

    # abrir os arquivos e pre-processa-los, ao fim add numa lista para depois ser modelado
    for file in files:
        with open(os.path.join(path_dir, file), 'r', encoding='utf-8') as f:
            words = nlp.prepoc_tokenizer(f.read())
            texts.append(words)

    # cria bigrams baseado nos textos processados
    bigram = Phrases(texts, min_count=5, threshold=10.0)
    texts = [bigram[txt] for txt in texts]

    # salva dataset preprocessado em csv
    with open(f'dataset/dataset_{path_dir.split("/")[-1]}.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(texts)

    return texts

if __name__ == '__main__':
    PATH = 'data_raw'
    FOLDER = 'data_pi'

    create_dataset(path_dir=f'{PATH}/{FOLDER}')
