"""
Data Downloading

!pip install transformers datasets tokenizers
!wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
!unzip -qq cornell_movie_dialogs_corpus.zip
!rm cornell_movie_dialogs_corpus.zip
!mkdir datasets
!mv cornell\ movie-dialogs\ corpus/movie_conversations.txt ./datasets
!mv cornell\ movie-dialogs\ corpus/movie_lines.txt ./datasets
"""

import os
from pathlib import Path
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm


def load_data():
    MAX_LEN = 64

    ### loading all data into memory
    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
        conv = c.readlines()
    with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
        lines = l.readlines()


    ### splitting text using special lines
    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    ### generate question answer pairs
    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids)):
            qa_pairs = []

            if i == len(ids) - 1:
                break

            first = lines_dic[ids[i]].strip()
            second = lines_dic[ids[i + 1]].strip()

            qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
            qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
            pairs.append(qa_pairs)

    return lines_dic, pairs


def wordpiece_tokenizer():
    ### save data as txt file
    data_path = "./data"
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    text_data = []
    file_count = 0

    lines_dic, pairs = load_data()

    for sample in tqdm.tqdm([x[0] for x in pairs]):
        text_data.append(sample)

        # once we hit the 10K mark, save to file
        if len(text_data) == 10000:
            with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    paths = [str(x) for x in Path('./data').glob('**/*.txt')]

    ### training own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

    tokenizer_path = "./bert-it-1"
    if not os.path.exists(tokenizer_path):
        os.mkdir('./bert-it-1')
    tokenizer.save_model('./bert-it-1', 'bert-it')
    tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

    return tokenizer
